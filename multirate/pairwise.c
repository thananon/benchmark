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
int pair_mode = 0;
char *buffer;

double g_start,g_end;

static pthread_barrier_t barrier;

void Test_Singlethreaded(void);
int Test_Multithreaded(void);


typedef struct{
    int id;
    int my_pair;
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
    while((c = getopt(argc,argv, "n:m:s:i:w:D:x:y:p")) != -1){
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
            case 'p':
                pair_mode = 1;
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
            printf("=================# MPI_THREAD_MULTIPLE mode #===================\n");

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



        /** if(thread_level != MPI_THREAD_MULTIPLE){ */
        /**     Test_Singlethreaded(); */
        /** } */
        /** else { */
            Test_Multithreaded();
        /** } */

       MPI_Finalize();

}

void *thread_work(void *info){

        char *buffer;
        int i,j,k,iteration;

        thread_info *t_info = (thread_info*) info;
        int tid = t_info->id;


        buffer = (char*) malloc(msg_size);

        pthread_barrier_wait(&barrier);

        if(i_am_sender){
        thread_info *t_info = (thread_info*) info;
        int tid = t_info->id;
            int total_request = window_size;
            MPI_Request request[ total_request ];
            MPI_Status status [ total_request ];

            for(iteration = 0; iteration < iter_num + warmup_num; iteration++){

                /* Basically we start taking time after some number of runs. */
                if(iteration == warmup_num){
                    pthread_barrier_wait(&barrier);
                    if(tid ==0 && me ==0) g_start = MPI_Wtime();
                }

                for(k=0;k<window_size;k++){
                    MPI_Isend(buffer, msg_size, MPI_BYTE, t_info->my_pair, 0, t_info->comm , &request[k]);
                }
                MPI_Waitall(total_request, request, status);

                if(tid==0){
                    for(i=0;i<y_recv_thread;i++){
                        MPI_Recv(buffer, 1, MPI_BYTE, t_info->my_pair, i, MPI_COMM_WORLD, &status[0]);
                     }
                }
            }
        }
        else{

            int total_request = window_size;
            MPI_Request request[ total_request ];
            MPI_Status status [ total_request ];

            for(iteration = 0; iteration < iter_num + warmup_num; iteration++){
                /* Basically we start taking time after some number of runs. */
                if(iteration == warmup_num){
                    pthread_barrier_wait(&barrier);
                    //start = MPI_Wtime();
                }

                /* post irecv to each reciever thread on each reciever. */
                for(k=0;k<window_size;k++){
                        MPI_Irecv(buffer, msg_size, MPI_BYTE, t_info->my_pair, 0, t_info->comm , &request[k]);
                }
                MPI_Waitall(total_request, request, status);

                MPI_Send(buffer, 1, MPI_BYTE, t_info->my_pair, tid, MPI_COMM_WORLD);

                /* We have to send something back to tell that we are done recving. */
                /** if(tid ==0){ */
                /**     for(i=0;i<n_send_process;i++){ */
                /**         MPI_Send(buffer, 1, MPI_BYTE, i, me, t_info->comm); */
                /**     } */
                /** } */
            }
        }

}

int Test_Multithreaded(void){

        pthread_t *id;
        thread_info *t_info;
        int i;
        int num_threads;
        /* Spawn the threads */
        if(i_am_sender)
            num_threads = x_send_thread;
        else
            num_threads = y_recv_thread;

        if(pair_mode){
            if( x_send_thread != y_recv_thread ){
                printf("number of send and recv threads should be the same.\n");
                return 0;
            }


        }

        id = (pthread_t*) malloc(sizeof(pthread_t) * num_threads);
        t_info = (thread_info*) malloc (sizeof(thread_info) * num_threads);
        pthread_barrier_init(&barrier, NULL, num_threads);


        for(i=0;i<num_threads;i++){
            t_info[i].id = i;
            t_info[i].my_pair = me-n_send_process;
            MPI_Comm_dup(MPI_COMM_WORLD, &t_info[i].comm);
            if(i_am_sender)
                t_info[i].my_pair = me+n_send_process;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        for(i=0;i<num_threads;i++){
            pthread_create(&id[i], NULL, thread_work, (void*) &t_info[i]);
        }
        // sync

        for(i=0;i<num_threads;i++){
            pthread_join(id[i], NULL);
        }
        if(me == 0)
        printf("rate %lf msg/s\n", (double)(msg_size*iter_num*window_size*num_threads/( MPI_Wtime() - g_start)));
        MPI_Barrier(MPI_COMM_WORLD);
        pthread_barrier_destroy(&barrier);
        return 0;
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
           /** printf("%d -> %lf\n",me, (double)msg_size*window_size*iter_num*m_recv_process/total_time); */

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
           /** printf("%d -> %lf\n",me, (double)msg_size*window_size*iter_num/total_time); */

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

