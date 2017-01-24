#include<stdio.h>
#include<mpi.h>
#include<stdlib.h>
#include<unistd.h>

int warmup_num = 10;
int window_size = 256;
int msg_size = 1024;
int iter_num = 100;
int me;
int size;


int n_send_process, m_recv_process;

char *buffer;

void process_args(int argc, char **argv){

    int c;
    while((c = getopt(argc,argv, "n:m:s:i:w:")) != -1){
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
            default:
                c = -1;
                exit(-1);
        }
    }
}

int main(int argc,char **argv){

        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &me);

        process_args(argc, argv);

        int i_am_sender = (me < n_send_process);

        buffer = (char*)malloc(msg_size);
        MPI_Request *request;
        MPI_Status *status;
        double start,total_time;

        if (i_am_sender){

           /* Warm up routine, this is the same as the real test but shorter. */
           /* "And that's what I said." */
           int i,j,k;
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
           printf("%d -> %lf\n",me, (double)msg_size*window_size*iter_num*m_recv_process*n_send_process/total_time);

        }
        else{

           /* This is the receiver side, we do everyting opposite from the sender */
           /* Warmup routine - again it is the same as real test but shorter. */
           int i,j,k;
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
            printf("Aggregated msg rate : %lf\n",(double)msg_size*window_size*iter_num*n_send_process*m_recv_process/(MPI_Wtime()-start));
            printf("total time = %lf\n",MPI_Wtime()-start);
        }
        MPI_Finalize();

}
