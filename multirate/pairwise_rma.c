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
int n_send_process = 1, m_recv_process = 1;
int x_send_thread=1,y_recv_thread=1;
int i_am_sender;
int pair_mode = 0;
char *buffer;
double g_start,g_end;
char *local_buffer,*shared_buffer;

MPI_Win win;

static pthread_barrier_t barrier;

void displayHelp(){

    printf("Pairwise performs ping/ping test between pairs. Which can be processes or threads and measure aggregated injection rate.\n");
    printf("In case of multiple process, please map the process by node. ie: even ranks are sender, odd ranks are receivers.\n");
    printf("Usage : ./pairwise [options]\n");
    printf("-s [n]       : message size of n bytes to send in the test.\n");
    printf("-p [n]       : number of process pairs in the test.\n");
    printf("-t [n]       : number of thread pairs in the test. \n");
    printf("-w [n]       : window size, number of posted message per iteration.\n");
    printf("-i [n]       : number of iterations.\n");
    printf("-h           : display this help message.\n");
    exit(0);
}

int perform_test(void);


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
    while((c = getopt(argc,argv, "s:i:w:D:t:hw:")) != -1){
        switch (c){
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
                    x_send_thread = 1;
                    y_recv_thread = 1;
                }
                break;
            case 't':
                y_recv_thread = atoi(optarg);
                x_send_thread = atoi(optarg);
                break;
            case 'h':
                displayHelp();
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

        if(size %2 != 0){
            if(me ==0) printf("Please use even number of processes.\n");
            MPI_Finalize();
            return 0;
        }

        n_send_process = m_recv_process = size/2;
        i_am_sender = 1 - (me % 2);

        local_buffer = (char*)malloc(sizeof(char)*msg_size);

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
       perform_test();
       MPI_Finalize();

}

void *thread_work(void *info){

        int i,j,k,iteration;

        thread_info *t_info = (thread_info*) info;
        int tid = t_info->id;

        pthread_barrier_wait(&barrier);
        MPI_Win_fence(0, win);
        int flag;
        if(i_am_sender){
            thread_info *t_info = (thread_info*) info;
            int tid = t_info->id;
            int total_request = window_size;

            for(iteration = 0; iteration < iter_num + warmup_num; iteration++){

                /* Basically we start taking time after some number of runs. */
                if(iteration == warmup_num){
                    pthread_barrier_wait(&barrier);
                    if(tid ==0 && me ==0) g_start = MPI_Wtime();
                }

                for(k=0;k<window_size;k++){
                    MPI_Put(local_buffer, msg_size, MPI_CHAR, t_info->my_pair, 0, msg_size, MPI_CHAR, win);
                }
                MPI_Win_fence(0, win);
            }
        }
        else{
            for(iteration = 0; iteration < iter_num + warmup_num; iteration++){
                MPI_Win_fence(0, win);
            }

        }

}

int perform_test(void){

        pthread_t *id;
        thread_info *t_info;
        int i;
        int num_threads;
        /* Spawn the threads */
        if(i_am_sender)
            num_threads = x_send_thread;
        else
            num_threads = y_recv_thread;

        if(pair_mode || 1){
            if( x_send_thread != y_recv_thread ){
                printf("number of send and recv threads should be the same.\n");
                return 0;
            }
        }

        id = (pthread_t*) malloc(sizeof(pthread_t) * num_threads);
        t_info = (thread_info*) malloc (sizeof(thread_info) * num_threads);
        pthread_barrier_init(&barrier, NULL, num_threads);


        local_buffer = (char*)malloc(sizeof(char)*msg_size);
        shared_buffer = (char*)malloc(sizeof(char)*msg_size);
        MPI_Win_create(shared_buffer, msg_size, sizeof(char), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        for(i=0;i<num_threads;i++){
            t_info[i].id = i;
            t_info[i].my_pair = me-1;
            MPI_Comm_dup(MPI_COMM_WORLD, &t_info[i].comm);
            if(i_am_sender)
                t_info[i].my_pair = me+1;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        for(i=0;i<num_threads;i++){
            pthread_create(&id[i], NULL, thread_work, (void*) &t_info[i]);
        }
        // sync

        for(i=0;i<num_threads;i++){
            pthread_join(id[i], NULL);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Win_free(&win);

        if(me == 0)
            printf("%d\t%d\t%d\t%d\t\t%lf\n",n_send_process, x_send_thread,window_size/256,
                                            msg_size,
                                            (double)((size/2)*iter_num*window_size*num_threads/( MPI_Wtime() - g_start)));
        pthread_barrier_destroy(&barrier);
        return 0;
}



