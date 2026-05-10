#include <optional>

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace {

static StringRef canonicalMPIName(StringRef N) {
  if (N.starts_with("PMPI_"))
    return N.drop_front(1); // PMPI_X -> MPI_X
  return N;
}

static bool isAnyMPIName(StringRef N) {
  N = canonicalMPIName(N);
  return N.starts_with("MPI_");
}

static bool isHelperName(StringRef N) {
  return N.starts_with("__mpiprog_");
}

struct AsyncStartDesc {
  unsigned CommArg;
  unsigned ReqArg;
};

static std::optional<AsyncStartDesc> getAsyncStartDesc(StringRef N) {
  N = canonicalMPIName(N);

  // P2P
  if (N == "MPI_Isend"  || N == "MPI_Irecv"  ||
      N == "MPI_Issend" || N == "MPI_Ibsend" ||
      N == "MPI_Irsend")
    return AsyncStartDesc{5, 6};

  // Collectives
  if (N == "MPI_Ibarrier")
    return AsyncStartDesc{0, 1};

  if (N == "MPI_Ibcast")
    return AsyncStartDesc{4, 5};

  if (N == "MPI_Ireduce")
    return AsyncStartDesc{6, 7};

  if (N == "MPI_Iallreduce")
    return AsyncStartDesc{5, 6};

  if (N == "MPI_Igather")
    return AsyncStartDesc{7, 8};

  if (N == "MPI_Iallgather")
    return AsyncStartDesc{6, 7};

  if (N == "MPI_Iscatter")
    return AsyncStartDesc{7, 8};

  if (N == "MPI_Ialltoall")
    return AsyncStartDesc{6, 7};

  if (N == "MPI_Iscan" || N == "MPI_Iexscan")
    return AsyncStartDesc{5, 6};

  // Common v-variants
  if (N == "MPI_Igatherv")
    return AsyncStartDesc{8, 9};

  if (N == "MPI_Iallgatherv")
    return AsyncStartDesc{7, 8};

  if (N == "MPI_Iscatterv")
    return AsyncStartDesc{8, 9};

  if (N == "MPI_Ialltoallv")
    return AsyncStartDesc{6, 7};

  return std::nullopt;
}

enum class CompletionKind {
  Wait,
  Test,
  Waitall,
  Testall,
  Waitany,
  Testany,
  Waitsome,
  Testsome,
};

struct CompletionDesc {
  CompletionKind Kind;
};

static std::optional<CompletionDesc> getCompletionDesc(StringRef N) {
  N = canonicalMPIName(N);

  if (N == "MPI_Wait")
    return CompletionDesc{CompletionKind::Wait};

  if (N == "MPI_Test")
    return CompletionDesc{CompletionKind::Test};

  if (N == "MPI_Waitall")
    return CompletionDesc{CompletionKind::Waitall};

  if (N == "MPI_Testall")
    return CompletionDesc{CompletionKind::Testall};

  if (N == "MPI_Waitany")
    return CompletionDesc{CompletionKind::Waitany};

  if (N == "MPI_Testany")
    return CompletionDesc{CompletionKind::Testany};

  if (N == "MPI_Waitsome")
    return CompletionDesc{CompletionKind::Waitsome};

  if (N == "MPI_Testsome")
    return CompletionDesc{CompletionKind::Testsome};

  return std::nullopt;
}

static bool isCandidateCallsite(CallBase *CB) {
  if (!CB || CB->isInlineAsm())
    return false;

  if (isa<IntrinsicInst>(CB))
    return false;

  Function *Callee = CB->getCalledFunction();
  if (!Callee)
    return true; // indirect call: keep as candidate

  StringRef N = Callee->getName();
  if (isAnyMPIName(N) || isHelperName(N))
    return false;

  return true;
}

static void collectLeafLoops(Loop *L, SmallVectorImpl<Loop *> &Out) {
  if (L->getSubLoops().empty()) {
    Out.push_back(L);
    return;
  }

  for (Loop *SubL : L->getSubLoops())
    collectLeafLoops(SubL, Out);
}

static void collectLeafLoops(LoopInfo &LI, SmallVectorImpl<Loop *> &Out) {
  for (Loop *TopL : LI)
    collectLeafLoops(TopL, Out);
}

static bool functionContainsAsyncMPIStart(Function &F) {
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (!CI)
        continue;

      Function *Callee = CI->getCalledFunction();
      if (!Callee)
        continue;

      if (getAsyncStartDesc(Callee->getName()))
        return true;
    }
  }
  return false;
}

class MPIAsyncProgressPass : public PassInfoMixin<MPIAsyncProgressPass> {
  struct CompletionSite {
    CallInst *CI = nullptr;
    CompletionKind Kind;
  };

  static FunctionCallee getMaybePollDecl(Module &M) {
    LLVMContext &Ctx = M.getContext();
    return M.getOrInsertFunction(
        "__mpiprog_maybe_poll",
        FunctionType::get(Type::getVoidTy(Ctx), false));
  }

  static FunctionCallee getAsyncStartDecl(Module &M, Type *ReqPtrTy, Type *CommTy) {
    LLVMContext &Ctx = M.getContext();
    return M.getOrInsertFunction(
        "__mpiprog_async_start",
        FunctionType::get(Type::getVoidTy(Ctx), {ReqPtrTy, CommTy}, false));
  }

  static FunctionCallee getAfterWaitDecl(Module &M, Type *ReqPtrTy) {
    LLVMContext &Ctx = M.getContext();
    return M.getOrInsertFunction(
        "__mpiprog_after_wait",
        FunctionType::get(Type::getVoidTy(Ctx), {ReqPtrTy}, false));
  }

  static FunctionCallee getAfterTestDecl(Module &M, Type *ReqPtrTy, Type *FlagPtrTy) {
    LLVMContext &Ctx = M.getContext();
    return M.getOrInsertFunction(
        "__mpiprog_after_test",
        FunctionType::get(Type::getVoidTy(Ctx), {ReqPtrTy, FlagPtrTy}, false));
  }

  static FunctionCallee getAfterWaitallDecl(Module &M, Type *CountTy, Type *ReqArrayTy) {
    LLVMContext &Ctx = M.getContext();
    return M.getOrInsertFunction(
        "__mpiprog_after_waitall",
        FunctionType::get(Type::getVoidTy(Ctx), {CountTy, ReqArrayTy}, false));
  }

  static FunctionCallee getAfterTestallDecl(Module &M, Type *CountTy, Type *ReqArrayTy,
                                            Type *FlagPtrTy) {
    LLVMContext &Ctx = M.getContext();
    return M.getOrInsertFunction(
        "__mpiprog_after_testall",
        FunctionType::get(Type::getVoidTy(Ctx), {CountTy, ReqArrayTy, FlagPtrTy}, false));
  }

  static FunctionCallee getAfterWaitanyDecl(Module &M, Type *CountTy, Type *ReqArrayTy,
                                            Type *IndexPtrTy) {
    LLVMContext &Ctx = M.getContext();
    return M.getOrInsertFunction(
        "__mpiprog_after_waitany",
        FunctionType::get(Type::getVoidTy(Ctx), {CountTy, ReqArrayTy, IndexPtrTy}, false));
  }

  static FunctionCallee getAfterTestanyDecl(Module &M, Type *CountTy, Type *ReqArrayTy,
                                            Type *IndexPtrTy, Type *FlagPtrTy) {
    LLVMContext &Ctx = M.getContext();
    return M.getOrInsertFunction(
        "__mpiprog_after_testany",
        FunctionType::get(Type::getVoidTy(Ctx),
                          {CountTy, ReqArrayTy, IndexPtrTy, FlagPtrTy}, false));
  }

  static FunctionCallee getAfterWaitsomeDecl(Module &M, Type *CountTy, Type *ReqArrayTy,
                                             Type *OutCountPtrTy, Type *IndicesPtrTy) {
    LLVMContext &Ctx = M.getContext();
    return M.getOrInsertFunction(
        "__mpiprog_after_waitsome",
        FunctionType::get(Type::getVoidTy(Ctx),
                          {CountTy, ReqArrayTy, OutCountPtrTy, IndicesPtrTy}, false));
  }

  static FunctionCallee getAfterTestsomeDecl(Module &M, Type *CountTy, Type *ReqArrayTy,
                                             Type *OutCountPtrTy, Type *IndicesPtrTy) {
    LLVMContext &Ctx = M.getContext();
    return M.getOrInsertFunction(
        "__mpiprog_after_testsome",
        FunctionType::get(Type::getVoidTy(Ctx),
                          {CountTy, ReqArrayTy, OutCountPtrTy, IndicesPtrTy}, false));
  }

public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    if (F.isDeclaration())
      return PreservedAnalyses::all();

    if (isHelperName(F.getName()))
      return PreservedAnalyses::all();

    Module &M = *F.getParent();

    SmallVector<CallInst *, 16> AsyncStarts;
    SmallVector<CompletionSite, 16> Completions;

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI)
          continue;

        Function *Callee = CI->getCalledFunction();
        if (!Callee)
          continue;

        if (getAsyncStartDesc(Callee->getName())) {
          AsyncStarts.push_back(CI);
          continue;
        }

        if (auto C = getCompletionDesc(Callee->getName())) {
          Completions.push_back({CI, C->Kind});
        }
      }
    }

    bool HasAsyncStart = !AsyncStarts.empty();
    if (!HasAsyncStart && Completions.empty())
      return PreservedAnalyses::all();

    bool Changed = false;

    // ------------------------------------------------------------
    // 1. Insert request start tracking
    // ------------------------------------------------------------
    for (CallInst *CI : AsyncStarts) {
      Function *Callee = CI->getCalledFunction();
      auto D = getAsyncStartDesc(Callee->getName());
      if (!D)
        continue;

      Value *ReqArg  = CI->getArgOperand(D->ReqArg);
      Value *CommArg = CI->getArgOperand(D->CommArg);

      Instruction *After = CI->getNextNode();
      if (!After)
        continue;

      IRBuilder<> B(After);
      auto Decl = getAsyncStartDecl(M, ReqArg->getType(), CommArg->getType());
      B.CreateCall(Decl, {ReqArg, CommArg});
      Changed = true;
    }

    // ------------------------------------------------------------
    // 2. Insert completion tracking
    // ------------------------------------------------------------
    for (const CompletionSite &S : Completions) {
      CallInst *CI = S.CI;
      Instruction *After = CI->getNextNode();
      if (!After)
        continue;

      IRBuilder<> B(After);

      switch (S.Kind) {
      case CompletionKind::Wait: {
        Value *ReqPtr = CI->getArgOperand(0);
        auto Decl = getAfterWaitDecl(M, ReqPtr->getType());
        B.CreateCall(Decl, {ReqPtr});
        Changed = true;
        break;
      }

      case CompletionKind::Test: {
        Value *ReqPtr  = CI->getArgOperand(0);
        Value *FlagPtr = CI->getArgOperand(1);
        auto Decl = getAfterTestDecl(M, ReqPtr->getType(), FlagPtr->getType());
        B.CreateCall(Decl, {ReqPtr, FlagPtr});
        Changed = true;
        break;
      }

      case CompletionKind::Waitall: {
        Value *Count = CI->getArgOperand(0);
        Value *Reqs  = CI->getArgOperand(1);
        auto Decl = getAfterWaitallDecl(M, Count->getType(), Reqs->getType());
        B.CreateCall(Decl, {Count, Reqs});
        Changed = true;
        break;
      }

      case CompletionKind::Testall: {
        Value *Count   = CI->getArgOperand(0);
        Value *Reqs    = CI->getArgOperand(1);
        Value *FlagPtr = CI->getArgOperand(2);
        auto Decl = getAfterTestallDecl(M, Count->getType(), Reqs->getType(),
                                        FlagPtr->getType());
        B.CreateCall(Decl, {Count, Reqs, FlagPtr});
        Changed = true;
        break;
      }

      case CompletionKind::Waitany: {
        Value *Count    = CI->getArgOperand(0);
        Value *Reqs     = CI->getArgOperand(1);
        Value *IndexPtr = CI->getArgOperand(2);
        auto Decl = getAfterWaitanyDecl(M, Count->getType(), Reqs->getType(),
                                        IndexPtr->getType());
        B.CreateCall(Decl, {Count, Reqs, IndexPtr});
        Changed = true;
        break;
      }

      case CompletionKind::Testany: {
        Value *Count    = CI->getArgOperand(0);
        Value *Reqs     = CI->getArgOperand(1);
        Value *IndexPtr = CI->getArgOperand(2);
        Value *FlagPtr  = CI->getArgOperand(3);
        auto Decl = getAfterTestanyDecl(M, Count->getType(), Reqs->getType(),
                                        IndexPtr->getType(), FlagPtr->getType());
        B.CreateCall(Decl, {Count, Reqs, IndexPtr, FlagPtr});
        Changed = true;
        break;
      }

      case CompletionKind::Waitsome: {
        Value *Count       = CI->getArgOperand(0);
        Value *Reqs        = CI->getArgOperand(1);
        Value *OutCountPtr = CI->getArgOperand(2);
        Value *IndicesPtr  = CI->getArgOperand(3);
        auto Decl = getAfterWaitsomeDecl(M, Count->getType(), Reqs->getType(),
                                         OutCountPtr->getType(), IndicesPtr->getType());
        B.CreateCall(Decl, {Count, Reqs, OutCountPtr, IndicesPtr});
        Changed = true;
        break;
      }

      case CompletionKind::Testsome: {
        Value *Count       = CI->getArgOperand(0);
        Value *Reqs        = CI->getArgOperand(1);
        Value *OutCountPtr = CI->getArgOperand(2);
        Value *IndicesPtr  = CI->getArgOperand(3);
        auto Decl = getAfterTestsomeDecl(M, Count->getType(), Reqs->getType(),
                                         OutCountPtr->getType(), IndicesPtr->getType());
        B.CreateCall(Decl, {Count, Reqs, OutCountPtr, IndicesPtr});
        Changed = true;
        break;
      }
      }
    }

    // ------------------------------------------------------------
    // 3. Insert maybe-poll sites in functions with async starts
    // ------------------------------------------------------------
    if (HasAsyncStart) {
      SmallVector<Instruction *, 32> PollSites;
      SmallPtrSet<Instruction *, 32> Seen;

      // Leaf loop headers
      LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
      SmallVector<Loop *, 16> LeafLoops;
      collectLeafLoops(LI, LeafLoops);

      for (Loop *L : LeafLoops) {
        BasicBlock *Header = L->getHeader();
        Instruction *IP = &*Header->getFirstInsertionPt();
        if (Seen.insert(IP).second)
          PollSites.push_back(IP);
      }

      // Non-MPI callsites
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          auto *CB = dyn_cast<CallBase>(&I);
          if (!CB)
            continue;

          if (!isCandidateCallsite(CB))
            continue;

          if (Seen.insert(&I).second)
            PollSites.push_back(&I);
        }
      }

      // Returns
      for (BasicBlock &BB : F) {
        Instruction *Term = BB.getTerminator();
        if (isa<ReturnInst>(Term)) {
          if (Seen.insert(Term).second)
            PollSites.push_back(Term);
        }
      }

      if (!PollSites.empty()) {
        FunctionCallee MaybePoll = getMaybePollDecl(M);
        for (Instruction *IP : PollSites) {
          IRBuilder<> B(IP);
          B.CreateCall(MaybePoll);
        }
        Changed = true;
      }
    }

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

static void addMPIAsyncProgressPipeline(ModulePassManager &MPM) {
  FunctionPassManager FPM;
  FPM.addPass(MPIAsyncProgressPass());
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
}

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {
      LLVM_PLUGIN_API_VERSION, "MPIAsyncProgress", "0.2",
      [](PassBuilder &PB) {
        PB.registerPipelineParsingCallback(
            [](StringRef Name, ModulePassManager &MPM,
               ArrayRef<PassBuilder::PipelineElement>) {
              if (Name == "mpi-async-progress") {
                addMPIAsyncProgressPipeline(MPM);
                return true;
              }
              return false;
            });

        PB.registerOptimizerEarlyEPCallback(
            [](ModulePassManager &MPM, OptimizationLevel,
               ThinOrFullLTOPhase Phase) {
              if (Phase == ThinOrFullLTOPhase::ThinLTOPreLink ||
                  Phase == ThinOrFullLTOPhase::FullLTOPreLink)
                return;

              addMPIAsyncProgressPipeline(MPM);
            });
      }};
}

