#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef MPIPROG_MAX_ACTIVE
#define MPIPROG_MAX_ACTIVE 8192
#endif

#ifndef MPIPROG_MAX_COMM_SNAPSHOT
#define MPIPROG_MAX_COMM_SNAPSHOT 16
#endif

typedef struct {
  MPI_Request *req_ptr;
  MPI_Comm comm;
} MPIPROG_Entry;

static MPIPROG_Entry g_entries[MPIPROG_MAX_ACTIVE];
static int g_entry_count = 0;

static size_t g_rr_cursor = 0;
static size_t g_replace_cursor = 0;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static int g_configured = 0;
static int g_enabled = 1;
static int g_period = 64;
static int g_comm_budget = 4;
static int g_debug = 0;
static int g_overflow_warned = 0;

static _Thread_local unsigned g_tls_counter = 0;

static void mpiprog_configure_once(void) {
  if (g_configured)
    return;

  const char *e;

  e = getenv("MPI_ASYNC_PROGRESS_ENABLE");
  if (e)
    g_enabled = (atoi(e) != 0);

  e = getenv("MPI_ASYNC_PROGRESS_PERIOD");
  if (e) {
    int p = atoi(e);
    if (p > 0)
      g_period = p;
  }

  e = getenv("MPI_ASYNC_PROGRESS_COMM_BUDGET");
  if (e) {
    int b = atoi(e);
    if (b > 0)
      g_comm_budget = b;
  }

  if (g_comm_budget > MPIPROG_MAX_COMM_SNAPSHOT)
    g_comm_budget = MPIPROG_MAX_COMM_SNAPSHOT;

  e = getenv("MPI_ASYNC_PROGRESS_DEBUG");
  if (e)
    g_debug = (atoi(e) != 0);

  g_configured = 1;
}

static int mpiprog_mpi_usable(void) {
  int init = 0;
  MPI_Initialized(&init);
  if (!init)
    return 0;

  int fin = 0;
  MPI_Finalized(&fin);
  if (fin)
    return 0;

  return 1;
}

static int mpiprog_find_req_ptr(MPI_Request *req_ptr) {
  int i;
  for (i = 0; i < g_entry_count; ++i) {
    if (g_entries[i].req_ptr == req_ptr)
      return i;
  }
  return -1;
}

static void mpiprog_remove_at(int idx) {
  if (idx < 0 || idx >= g_entry_count)
    return;

  g_entries[idx] = g_entries[g_entry_count - 1];
  g_entry_count--;

  if (g_rr_cursor >= (size_t)g_entry_count)
    g_rr_cursor = 0;
}

static void mpiprog_remove_req_ptr_unlocked(MPI_Request *req_ptr) {
  int idx;
  if (!req_ptr)
    return;

  idx = mpiprog_find_req_ptr(req_ptr);
  if (idx >= 0)
    mpiprog_remove_at(idx);
}

static void mpiprog_add_or_update_unlocked(MPI_Request *req_ptr, MPI_Comm comm) {
  int idx;

  if (!req_ptr)
    return;

  idx = mpiprog_find_req_ptr(req_ptr);
  if (idx >= 0) {
    g_entries[idx].comm = comm;
    return;
  }

  if (g_entry_count < MPIPROG_MAX_ACTIVE) {
    g_entries[g_entry_count].req_ptr = req_ptr;
    g_entries[g_entry_count].comm = comm;
    g_entry_count++;
    return;
  }

  if (!g_overflow_warned) {
    g_overflow_warned = 1;
    fprintf(stderr,
            "[MPIAsyncProgress] warning: active request registry overflow "
            "(capacity=%d). Replacing old entries.\n",
            MPIPROG_MAX_ACTIVE);
    fflush(stderr);
  }

  idx = (int)(g_replace_cursor % (size_t)MPIPROG_MAX_ACTIVE);
  g_replace_cursor++;
  g_entries[idx].req_ptr = req_ptr;
  g_entries[idx].comm = comm;
}

static int mpiprog_comm_in_list(MPI_Comm comms[], int n, MPI_Comm c) {
  int i;
  for (i = 0; i < n; ++i) {
    if (comms[i] == c)
      return 1;
  }
  return 0;
}

static int mpiprog_snapshot_comms(MPI_Comm out[], int max_out) {
  int n = 0;
  int scanned = 0;
  size_t start;

  if (g_entry_count == 0 || max_out <= 0)
    return 0;

  start = g_rr_cursor;

  while (scanned < g_entry_count && n < max_out) {
    size_t idx = (start + (size_t)scanned) % (size_t)g_entry_count;
    MPI_Comm c = g_entries[idx].comm;

    if (!mpiprog_comm_in_list(out, n, c))
      out[n++] = c;

    scanned++;
  }

  if (g_entry_count > 0)
    g_rr_cursor = (start + (size_t)scanned) % (size_t)g_entry_count;
  else
    g_rr_cursor = 0;

  return n;
}

static void mpiprog_remove_completed_from_array_unlocked(int count, MPI_Request reqs[]) {
  int i;
  if (!reqs || count <= 0)
    return;

  for (i = 0; i < count; ++i) {
    if (reqs[i] == MPI_REQUEST_NULL)
      mpiprog_remove_req_ptr_unlocked(&reqs[i]);
  }
}

static void mpiprog_remove_all_from_array_unlocked(int count, MPI_Request reqs[]) {
  int i;
  if (!reqs || count <= 0)
    return;

  for (i = 0; i < count; ++i)
    mpiprog_remove_req_ptr_unlocked(&reqs[i]);
}

void __mpiprog_async_start(MPI_Request *req_ptr, MPI_Comm comm) {
  mpiprog_configure_once();
  if (!g_enabled)
    return;
  if (!mpiprog_mpi_usable())
    return;

  pthread_mutex_lock(&g_lock);
  mpiprog_add_or_update_unlocked(req_ptr, comm);
  pthread_mutex_unlock(&g_lock);
}

void __mpiprog_after_wait(MPI_Request *req_ptr) {
  mpiprog_configure_once();
  if (!g_enabled)
    return;

  pthread_mutex_lock(&g_lock);
  mpiprog_remove_req_ptr_unlocked(req_ptr);
  pthread_mutex_unlock(&g_lock);
}

void __mpiprog_after_test(MPI_Request *req_ptr, int *flag_ptr) {
  int done = 0;

  mpiprog_configure_once();
  if (!g_enabled)
    return;

  if (flag_ptr && *flag_ptr)
    done = 1;
  else if (req_ptr && *req_ptr == MPI_REQUEST_NULL)
    done = 1;

  if (!done)
    return;

  pthread_mutex_lock(&g_lock);
  mpiprog_remove_req_ptr_unlocked(req_ptr);
  pthread_mutex_unlock(&g_lock);
}

void __mpiprog_after_waitall(int count, MPI_Request reqs[]) {
  mpiprog_configure_once();
  if (!g_enabled)
    return;

  pthread_mutex_lock(&g_lock);
  mpiprog_remove_all_from_array_unlocked(count, reqs);
  pthread_mutex_unlock(&g_lock);
}

void __mpiprog_after_testall(int count, MPI_Request reqs[], int *flag_ptr) {
  int done = 0;

  mpiprog_configure_once();
  if (!g_enabled)
    return;

  if (flag_ptr && *flag_ptr)
    done = 1;

  pthread_mutex_lock(&g_lock);
  if (done)
    mpiprog_remove_all_from_array_unlocked(count, reqs);
  else
    mpiprog_remove_completed_from_array_unlocked(count, reqs);
  pthread_mutex_unlock(&g_lock);
}

void __mpiprog_after_waitany(int count, MPI_Request reqs[], int *index_ptr) {
  int idx;

  mpiprog_configure_once();
  if (!g_enabled)
    return;

  if (!reqs || !index_ptr)
    return;

  idx = *index_ptr;
  if (idx == MPI_UNDEFINED || idx < 0 || idx >= count)
    return;

  pthread_mutex_lock(&g_lock);
  mpiprog_remove_req_ptr_unlocked(&reqs[idx]);
  pthread_mutex_unlock(&g_lock);
}

void __mpiprog_after_testany(int count, MPI_Request reqs[], int *index_ptr, int *flag_ptr) {
  int idx;
  int done = 0;

  mpiprog_configure_once();
  if (!g_enabled)
    return;

  if (flag_ptr && *flag_ptr)
    done = 1;

  if (!done || !reqs || !index_ptr)
    return;

  idx = *index_ptr;
  if (idx == MPI_UNDEFINED || idx < 0 || idx >= count)
    return;

  pthread_mutex_lock(&g_lock);
  mpiprog_remove_req_ptr_unlocked(&reqs[idx]);
  pthread_mutex_unlock(&g_lock);
}

void __mpiprog_after_waitsome(int incount, MPI_Request reqs[],
                              int *outcount_ptr, int indices[]) {
  int outcount, i;

  mpiprog_configure_once();
  if (!g_enabled)
    return;

  if (!reqs || !outcount_ptr || !indices)
    return;

  outcount = *outcount_ptr;
  if (outcount == MPI_UNDEFINED || outcount <= 0)
    return;

  pthread_mutex_lock(&g_lock);
  for (i = 0; i < outcount; ++i) {
    int idx = indices[i];
    if (idx >= 0 && idx < incount)
      mpiprog_remove_req_ptr_unlocked(&reqs[idx]);
  }
  pthread_mutex_unlock(&g_lock);
}

void __mpiprog_after_testsome(int incount, MPI_Request reqs[],
                              int *outcount_ptr, int indices[]) {
  int outcount, i;

  mpiprog_configure_once();
  if (!g_enabled)
    return;

  if (!reqs || !outcount_ptr || !indices)
    return;

  outcount = *outcount_ptr;
  if (outcount == MPI_UNDEFINED || outcount <= 0)
    return;

  pthread_mutex_lock(&g_lock);
  for (i = 0; i < outcount; ++i) {
    int idx = indices[i];
    if (idx >= 0 && idx < incount)
      mpiprog_remove_req_ptr_unlocked(&reqs[idx]);
  }
  pthread_mutex_unlock(&g_lock);
}

void __mpiprog_maybe_poll(void) {
  MPI_Comm comms[MPIPROG_MAX_COMM_SNAPSHOT];
  int ncomms, i;

  mpiprog_configure_once();
  if (!g_enabled)
    return;

  g_tls_counter++;
  if ((g_tls_counter % (unsigned)g_period) != 0)
    return;

  if (!mpiprog_mpi_usable())
    return;

  pthread_mutex_lock(&g_lock);
  ncomms = mpiprog_snapshot_comms(comms, g_comm_budget);
  pthread_mutex_unlock(&g_lock);

  if (ncomms <= 0)
    return;

  for (i = 0; i < ncomms; ++i) {
    int flag = 0;
    MPI_Status status;
    (void)MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comms[i], &flag, &status);
  }

  if (g_debug) {
    static _Thread_local unsigned dbg_counter = 0;
    dbg_counter++;
    if ((dbg_counter % 1024u) == 0u) {
      fprintf(stderr,
              "[MPIAsyncProgress] poll: probed %d communicator(s)\n",
              ncomms);
      fflush(stderr);
    }
  }
}

