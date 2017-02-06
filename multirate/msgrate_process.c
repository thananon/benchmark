#include<stdio.h>
#include<mpi.h>
#include<stdlib.h>
#include<unistd.h>
#include<assert.h>

int warmup_num = 0;
int window_size = 256;
int msg_size = 1024;
int iter_num = 1;
int want_multithread = 1;
int me;
int size;
int n_send_process, m_recv_process;
int x_send_thread=1,y_recv_thread=1;
int i_am_sender;
int num_comm;
char *buffer;
double g_start;

static pthread_barrier_t barrier;

void Test_Singlethreaded(void);
void Test_Multithreaded(void);

MPI_Comm *comm;

typedef struct{
    int id;
    MPI_Comm comm;
}thread_info;

void preprocess_args(int argc, char **argv){

    /* Copied this part from artem's benchmark. He said usually
     * we can't access argv before we call MPI_Init but by doing this,
     * it works. I dont know anything about this but it does work. */
    int i;
    for(i=0;i<argc;i++){
        if(!strcmp(argv[i], "-Dthrds"))
           want_multithread = 0;
    }

}

void process_args(int argc, char **argv){

    int c;
    while((c = getopt(argc,argv, "n:m:s:i:w:D:x:y:")) != -1){
        switch (c){
            case 'n':
                n_send_process = atoi(optarg);
                break;
            case 'm':
                m_recv_process = atoi(optarg);
                break;
            case 's':
                msg_size = atoi(optarg);
                break;
            case 'i':
                iter_num = atoi(optarg);
                break;
            case 'w':
                window_size = atoi(optarg);
                break;
            case 'D':
                if(!strcmp("thrds",optarg)){
                    want_multithread = 0;
                }
                break;
            case 'x':
                x_send_thread = atoi(optarg);
                break;
            case 'y':
                y_recv_thread = atoi(optarg);
                break;
            default:
                c = -1;
                exit(-1);
        }
    }
}

int main(int argc,char **argv){

        /* Process the arguments. */
        preprocess_args(argc, argv);
        int thread_level;

        /* Initialize MPI according to the user's desire.*/
        if(want_multithread){
            MPI_Init_thread(&argc,&argv, MPI_THREAD_MULTIPLE, &thread_level);
            if(thread_level != MPI_THREAD_MULTIPLE){
                printf("MPI_THREAD_MULTIPLE requested but MPI implementation cannot provide.\n");
                MPI_Finalize();
                return 0;
            }

        }
        else
            MPI_Init(&argc, &argv);


        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &me);

#ifdef GDB
        printf("Rank %d : PID %d\n",me,getpid());
        printf("Waiting for gdb attach.\n");

        int gdb = 0;
        if(me==1)
        while(!gdb){
            sleep(1);
        }

#endif

        /* Process the arguments. */
        process_args(argc,argv);

        if(size != n_send_process + m_recv_process){
            if(me == 0)
                printf("ERROR : number of process must be n+m\n");
            MPI_Finalize();
            return 0;
        }

        i_am_sender = (me < n_send_process);

        Test_Multithreaded();

        MPI_Finalize();

}

void *thread_work(void *info){

        char *buffer;
        int i,j,k,iteration;
        double start, end;

        thread_info *t_info = (thread_info*) info;
        int tid = t_info->id;


        buffer = (char*) malloc(msg_size);

        pthread_barrier_wait(&barrier);

        if(i_am_sender){
        thread_info *t_info = (thread_info*) info;
        int tid = t_info->id;
            int total_request = m_recv_process * y_recv_thread * window_size;
            MPI_Request request[ total_request ];
            MPI_Status status [ total_request ];

            for(iteration = 0; iteration < iter_num + warmup_num; iteration++){

                /* Basically we start taking time after some number of runs. */
                if(iteration == warmup_num){
                    pthread_barrier_wait(&barrier);
                    if(tid==0)
                        g_start = MPI_Wtime();
                }

                /* post isend to each reciever thread on each reciever. */
                for(i=0;i<m_recv_process;i++){
                    for(j=0;j<y_recv_thread;j++){
                        int offset = (i*y_recv_thread + j) * window_size;
                        int comm_offset = (me*x_send_thread*y_recv_thread) + (tid * y_recv_thread) + j;
                        for(k=0;k<window_size;k++){
                            MPI_Isend(buffer, msg_size, MPI_BYTE, i+n_send_process, j, comm[comm_offset] , &request[offset+k]);
                        }
                    }
                }
                MPI_Waitall(total_request, request, status);

                if(tid==0){
                    for(i=0;i<m_recv_process;i++){
                        MPI_Recv(buffer, 1, MPI_BYTE, i+n_send_process, i+n_send_process, MPI_COMM_WORLD, &status[0]);
                     }
                }
            }
        }
        else{

            int total_request = n_send_process * x_send_thread * window_size;
            MPI_Request request[ total_request ];
            MPI_Status status [ total_request ];

            for(iteration = 0; iteration < iter_num + warmup_num; iteration++){
                /* Basically we start taking time after some number of runs. */
                if(iteration == warmup_num){
                    pthread_barrier_wait(&barrier);
                    start = MPI_Wtime();
                }

                /* post irecv to each reciever thread on each reciever. */
                for(i=0;i<n_send_process;i++){
                    for(j=0;j<x_send_thread;j++){
                        int offset = (i*x_send_thread + j) * window_size;
                        int comm_offset = (i*x_send_thread*y_recv_thread) + (j * y_recv_thread) + tid;
                        for(k=0;k<window_size;k++){
                            MPI_Irecv(buffer, msg_size, MPI_BYTE, i, tid, comm[comm_offset] , &request[offset+k]);
                        }
                    }
                }
                MPI_Waitall(total_request, request, status);

                /* We have to send something back to tell that we are done recving. */
                if(tid ==0){
                    for(i=0;i<n_send_process;i++){
                        MPI_Send(buffer, 1, MPI_BYTE, i, me, MPI_COMM_WORLD);
                    }
                }
            }
        }

}

void Test_Multithreaded(void){

        pthread_t *id;
        thread_info *t_info;
        int i;
        int num_threads;

        num_comm = n_send_process * x_send_thread * m_recv_process * y_recv_thread;
        /* Spawn the threads */
        if(i_am_sender)
            num_threads = x_send_thread;
        else
            num_threads = y_recv_thread;

        id = (pthread_t*) malloc(sizeof(pthread_t) * num_threads);
        t_info = (thread_info*) malloc (sizeof(thread_info) * num_threads);
        pthread_barrier_init(&barrier, NULL, num_threads);
        comm = (MPI_Comm*) malloc(sizeof(MPI_Comm) * num_comm);

        for(i=0;i<num_comm;i++){
            MPI_Comm_dup(MPI_COMM_WORLD, &comm[i]);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        for(i=0;i<num_threads;i++){
            t_info[i].id = i;
            t_info[i].comm = MPI_COMM_WORLD;
            pthread_create(&id[i], NULL, thread_work, (void*) &t_info[i]);
        }

        // sync

        for(i=0;i<num_threads;i++){
            pthread_join(id[i], NULL);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        /* output message rate */
        if(me == 0){
            printf("%d\t%d\t%d\t%d\t\t%lf\n",n_send_process, x_send_thread
                                            , m_recv_process, y_recv_thread
                                            ,(double)(num_comm *window_size*iter_num/(MPI_Wtime() - g_start)));
        }
        pthread_barrier_destroy(&barrier);
}

void Test_Singlethreaded(void){

        int i,j,k;
        double start,total_time;
        buffer = (char*)malloc(msg_size);

        MPI_Request *request;
        MPI_Status *status;

        if (i_am_sender){

           /* Warm up routine, this is the same as the real test but shorter. */
           /* "And that's what I said." */
           request = (MPI_Request*)malloc(sizeof(MPI_Request)*window_size*m_recv_process);
           status = (MPI_Status*)malloc(sizeof(MPI_Status)*window_size*m_recv_process);
           for(i=0;i<warmup_num;i++){
                for(k=0; k < m_recv_process; k++){
                    for(j=0;j<window_size;j++){
                        MPI_Isend(&buffer, msg_size, MPI_BYTE, n_send_process+k, me, MPI_COMM_WORLD, &request[k*window_size+j]);
                    }
                }
                MPI_Waitall(window_size*m_recv_process, request, status);
           }

           /* Sync everyone, we are starting.*/
           /* We are posting bunch of sends to each receiver. */
           MPI_Barrier(MPI_COMM_WORLD);
           start = MPI_Wtime();

           for(i=0;i<iter_num;i++){
                for(k=0; k < m_recv_process; k++){
                    for(j=0;j<window_size;j++){
                        MPI_Isend(&buffer, msg_size, MPI_BYTE, n_send_process+k, me, MPI_COMM_WORLD, &request[k*window_size+j]);
                    }
                }
                MPI_Waitall(window_size*m_recv_process, request, status);
           }

           /* Get result. */
           total_time = MPI_Wtime() - start;
           printf("%d -> %lf\n",me, (double)msg_size*window_size*iter_num*m_recv_process/total_time);

        }
        else{

           /* This is the receiver side, we do everyting opposite from the sender */
           /* Warmup routine - again it is the same as real test but shorter. */
           request = (MPI_Request*)malloc(sizeof(MPI_Request)*window_size*n_send_process);
           status = (MPI_Status*)malloc(sizeof(MPI_Status)*window_size*n_send_process);

           for(i=0;i<warmup_num;i++){
                for(k=0; k < n_send_process; k++){
                    for(j=0;j<window_size;j++){
                        MPI_Irecv(&buffer, msg_size, MPI_BYTE, k, k, MPI_COMM_WORLD, &request[k*window_size+j]);
                    }
                }
                MPI_Waitall(window_size*n_send_process, request, status);
           }

           /* Sync everyone and start the test. */
           MPI_Barrier(MPI_COMM_WORLD);
           start = MPI_Wtime();

           for(i=0;i<iter_num;i++){
                for(k=0; k < n_send_process; k++){
                    for(j=0;j<window_size;j++){
                        MPI_Irecv(&buffer, msg_size, MPI_BYTE, k, k, MPI_COMM_WORLD, &request[k*window_size+j]);
                    }
                }
                MPI_Waitall(window_size*n_send_process, request, status);
           }

           /* Get result. */
           total_time = MPI_Wtime() - start;
           printf("%d -> %lf\n",me, (double)msg_size*window_size*iter_num/total_time);

        }

        /* Make sure everyone is here before we take the time for aggregated performance. */
        MPI_Barrier(MPI_COMM_WORLD);

        /* Show the result. I picked rank 0 out of ease of coding. */
        if(me == 0){
            printf("Aggregated msg rate : %lf\n",(double)msg_size *
                                                            window_size *
                                                            iter_num*n_send_process *
                                                            m_recv_process /
                                                            (MPI_Wtime()-start));
            printf("total time = %lf\n",MPI_Wtime()-start);
        }

}

