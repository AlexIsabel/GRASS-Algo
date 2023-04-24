/*
    pGRASS 实现
*/

#include <stack>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <math.h>
#include <sys/time.h>
#include <cstring>
#include <omp.h>
#include <immintrin.h>
#include <queue>


#define ROW 0
#define COLUMN 1
#define VALUE 2
#define vertex1 0
#define vertex2 1
#define Weight 2
#define Threads_sort 16

using namespace std;

typedef double DataType;

struct ReadData {
    int row;
    int col;
    DataType value;
};
struct Edge{
    int u, v;
    DataType w;
    DataType eff_w;
    DataType re_dist;
};
struct LG{
    int u;
    DataType w;
};
vector< vector <LG> > LG_vec;


inline bool compare_eff(const Edge a, const Edge b) {    return a.eff_w > b.eff_w;   }
inline bool compare_re(const Edge a, const Edge b) {    return a.re_dist > b.re_dist;   }

inline int findSet(int x, int* fa) {
    if (fa[x] != x) fa[x] = findSet(fa[x], fa);
    return fa[x];
}

inline void belta_bfs(int belta, bool* vis, int st, const int M)
{
    for (int j=0; j<M+1; j++) vis[j] = 0;
    queue<int> bfs1;

    int layer = 0;
    bfs1.push(st);    
    bfs1.push(0);
    vis[st] = 1;

    while (layer < belta) {
        int u = bfs1.front();
        bfs1.pop();
        if (u == 0) {  layer++; bfs1.push(0);   }
        else {
            for (int id=0; id<LG_vec[u].size(); id++) {
                int v = LG_vec[u][id].u;
                if (vis[v] == 0) {
                    bfs1.push(v);
                    vis[v] = 1;
                }
            }
        }
    }
}

int main(int argc, const char * argv[]) {
    // read input file
    const char* file = "";
    if(argc > 2) {
        printf("Usage : ./main <filename>");
        exit(0);
    } else if(argc == 2) {
        file = argv[1];
    }
    // the matrix you read must be a adjacency matrix
    ifstream fin(file);
    while (fin.peek() == '%') fin.ignore(2048, '\n');   // Ignore headers and comments

    // -------- read from file -----------
    int M, N, L;
    fin >> M >> N >> L;
    // DataType volume[M+1], degree[M+1];  //degree of every point
    DataType* volume = (DataType*) calloc(M+1, sizeof(DataType));
    DataType* degree = (DataType*) calloc(M+1, sizeof(DataType));
    for (int i=0; i<=M; i++) volume[i] = 0.0;
    for (int i=0; i<=M; i++) degree[i] = 0.0;

    // Big Data: vec vec 
    vector< vector<ReadData>> triple;
    triple.resize(M+1);

    int edge_cnt = 0;
    Edge* edge_mtx = (Edge*) calloc(L, sizeof(Edge));

    //fill matrix vector and calculate the degree and volume of every point
    int m, n = 0;
    DataType data = 0;
    for (int i = 0; i < L; ++i) {
        fin >> m >> n >> data;
        data = fabs(data);  // positive value
        edge_mtx[edge_cnt++] = Edge{m, n, data};
        volume[n] = volume[n] + data;
        degree[n] = degree[n] + 1;
        triple[n].emplace_back((ReadData){m, n, data});
        volume[m] = volume[m] + data;
        degree[m] = degree[m] + 1;
        triple[m].emplace_back((ReadData){n, m, data});
    }
    fin.close();

    /**************************************************/
    /***************** Start timing *******************/
    /**************************************************/
    struct timeval start, end;
    gettimeofday(&start, NULL);

    struct  timeval start1;
    gettimeofday(&start1, NULL);
    

    //1. largest volume point as root
    int largest_volume_point = 0;
    double largest_volume = 0;
    for (int i=1;i<=M;i++)
        if (volume[i] > largest_volume)
            largest_volume_point = i, largest_volume = volume[i];



    gettimeofday(&end, NULL);
    printf("1 root: \t\t %f ms\n", (end.tv_sec-start1.tv_sec)*1000+(end.tv_usec-start1.tv_usec)/1000.0);
    // printf("%f\n", (end.tv_sec-start1.tv_sec)*1000+(end.tv_usec-start1.tv_usec)/1000.0);
    gettimeofday(&start1, NULL);


    //2. BFS: no-weight distance
    // can be paralleled
    int* no_weight_distance = (int*) calloc(M+1, sizeof(int));
    for (int i=0; i<M+1; i++) no_weight_distance[i] = -1;
    queue<int> process;
    process.push(largest_volume_point);
    no_weight_distance[largest_volume_point] = 0;

    while (process.size()) {
        int point = process.front();
        process.pop();
        for(int i=0; i<triple[point].size(); i++){
            if(no_weight_distance[triple[point][i].row] == -1){
                process.push(triple[point][i].row);
                no_weight_distance[triple[point][i].row] = no_weight_distance[point] + 1;
            }
        }
    }


    gettimeofday(&end, NULL);
    printf("2 Bfs: \t\t\t %f ms\n", (end.tv_sec-start1.tv_sec)*1000+(end.tv_usec-start1.tv_usec)/1000.0);
    // printf("%f\n", (end.tv_sec-start1.tv_sec)*1000+(end.tv_usec-start1.tv_usec)/1000.0);
    gettimeofday(&start1, NULL);


    //3. Calculate g for MST each edge  
    // effective weight
    for(int i = 1; i < M+1; i++) degree[i] = log(degree[i]);
    #pragma omp parallel for
    for(int i = 0; i < edge_cnt; i++){
        int node1 = edge_mtx[i].u;
        int node2 = edge_mtx[i].v;
        double a = degree[node1];
        double b = degree[node2];
        double c = fmax(a, b);
        // double c = (a > b) ? a : b;
        double d = edge_mtx[i].w;
        double e = no_weight_distance[node1];
        double f = no_weight_distance[node2];
        // double g = d * log(c) / (f+e);
        double g = d * c / (f+e);
        edge_mtx[i].eff_w = g;
    }


    gettimeofday(&end, NULL);
    printf("3 Cal for edges(p): \t %f ms\n", (end.tv_sec-start1.tv_sec)*1000+(end.tv_usec-start1.tv_usec)/1000.0);
    // printf("%f\n", (end.tv_sec-start1.tv_sec)*1000+(end.tv_usec-start1.tv_usec)/1000.0);
    gettimeofday(&start1, NULL);


    //4. sort according effective weight of each edge
    stable_sort(edge_mtx, edge_mtx+edge_cnt, compare_eff);


    gettimeofday(&end, NULL);
    printf("4 sort(p): \t\t %f ms\n", (end.tv_sec-start1.tv_sec)*1000+(end.tv_usec-start1.tv_usec)/1000.0);
    //printf("%f\n", (end.tv_sec-start1.tv_sec)*1000+(end.tv_usec-start1.tv_usec)/1000.0);
    gettimeofday(&start1, NULL);



    //5. Kruskal on new weight & get off-tree edge set
    // can be paralleled
    int* assistance = (int*) calloc(M+1, sizeof(int));
    for (int i=0; i<=N; i++) assistance[i] = i;
    Edge* spanning_tree = (Edge*) calloc(int(L), sizeof(Edge));
    Edge* off_tree_edge = (Edge*) calloc(int(L), sizeof(Edge));
    int sp_tree_cnt = 0;
    int off_tree_cnt = 0;
    int k = 0;
    int stop_point = -1;
    for (int i=0; i<edge_cnt; i++) {
        int fa = findSet(edge_mtx[i].u, assistance);
        int fb = findSet(edge_mtx[i].v, assistance);
        if (fa != fb) {
            k++;
            spanning_tree[sp_tree_cnt++] = edge_mtx[i];
            if (fa < fb) assistance[fa] = fb;
            else assistance[fb] = fa;
        } else off_tree_edge[off_tree_cnt++] = edge_mtx[i];
        if (k == M-1) {
            for (int j=i+1; j<edge_cnt; j++)
                off_tree_edge[off_tree_cnt++] = edge_mtx[j];
            break;
        }
    }

    
    gettimeofday(&end, NULL);
    printf("5 MST: \t\t\t %f ms\n", (end.tv_sec-start1.tv_sec)*1000+(end.tv_usec-start1.tv_usec)/1000.0);
    // printf("%f\n", (end.tv_sec-start1.tv_sec)*1000+(end.tv_usec-start1.tv_usec)/1000.0);
    gettimeofday(&start1, NULL);


    
    //6. Get vec graph for MST
    // vector< vector <LG> > LG_vec;
    LG_vec.resize(M+1);
    for (int i=0; i<sp_tree_cnt; i++) {
        int u = spanning_tree[i].u;
        int v = spanning_tree[i].v;

        if (u > M || v > M) cout << "err\n";

        double wt = 1.0 / spanning_tree[i].w;
        LG_vec[u].emplace_back(LG{v, wt});
        LG_vec[v].emplace_back(LG{u, wt});
    }


    gettimeofday(&end, NULL);
    printf("6 Build MST vec: \t %f ms\n", (end.tv_sec-start1.tv_sec)*1000+(end.tv_usec-start1.tv_usec)/1000.0);
    //printf("%f\n", (end.tv_sec-start1.tv_sec)*1000+(end.tv_usec-start1.tv_usec)/1000.0);
    gettimeofday(&start1, NULL);



    //8. calculate the resistance of each off_tree edge
    // step1: get mst_depth
    int* mst_depth = (int*) calloc(M+1, sizeof(int));
    int* mst_root = (int*) calloc(M+1, sizeof(int));
    double* mst_dist = (double*) calloc(M+1, sizeof(double));
    for (int i=0; i<=M; i++) {
        mst_depth[i] = -1;
        mst_root[i] = -1;
        mst_dist[i] = 0.0;
    }
    const double gsubtree_portion = 0.2;        // the proportion of subtree
    int gsubtree_cnt = M * gsubtree_portion;

    queue<int> mst_bfs;
    int mst_max_depth = 0;
    int mst_gtree_depth = -1;   // global_subtree;

    mst_bfs.push(largest_volume_point);
    mst_depth[largest_volume_point] = 0;
    mst_root[largest_volume_point] = largest_volume_point;
    int g_cnt = 1;

    vector<int> subtree_roots;  // subtrees' root
    int subtree_cnt = 0;        // subtree num
    int subtree_depth = 3;

    while (mst_bfs.size())
    {
        int u = mst_bfs.front();
        mst_bfs.pop();
        int curr_depth = mst_depth[u];
        if (curr_depth == subtree_depth) break;

        for (int i=0; i<LG_vec[u].size(); i++) {
            int v = LG_vec[u][i].u;
            double w = LG_vec[u][i].w;
            if (mst_depth[v] == -1) {
                mst_depth[v] = mst_depth[u] + 1;
                mst_dist[v] += mst_dist[u] + w;
                mst_root[v] = largest_volume_point;
                g_cnt++;
                mst_bfs.push(v);
            }
        }
    }

    for (int i=1; i<=M; i++) 
        if (mst_depth[i] == subtree_depth) subtree_roots.emplace_back(i); 

    // parallel for subtrees
    #pragma omp prallel for
    for (int i=0; i<subtree_roots.size(); i++) {
        int r = subtree_roots[i];

        queue<int> sub_bfs;
        sub_bfs.push(r);
        // mst_root[r] = r;
        
        while (sub_bfs.size())
        {
            int u = sub_bfs.front();
            sub_bfs.pop();

            for (int j=0; j<LG_vec[u].size(); j++) {
                int v = LG_vec[u][j].u;
                double w = LG_vec[u][j].w;
                if (mst_depth[v] == -1) {
                    mst_depth[v] = mst_depth[u] + 1;
                    mst_dist[v] = mst_dist[u] + w;
                    mst_root[v] = r;
                    sub_bfs.push(v);
                }
            }
        }
    }


    // calculate LCA for re_distance 
    #pragma omp parallel for 
    for (int i=0; i<off_tree_cnt; i++) {
        int u = off_tree_edge[i].u;
        int v = off_tree_edge[i].v;
        double re_dist = 0.0;

        int node1, node2;
        if (mst_root[u] == mst_root[v])
            node1 = u, node2 = v;
        else {
            if (mst_root[u] != largest_volume_point) 
                node1 = mst_root[u];
            else node1 = u;
            if (mst_root[v] != largest_volume_point)
                node2 = mst_root[v];
            else node2 = v;
        }

        // cout << i << " ----- \n";
      
        // run targin to find dist
        stack<int> stk; //to show the process of dfs
        // int vis[M+1];  //to show whether a point has been gone through
        // int find[M+1];  //Joint search set
        // int position_node[M+1]; //restore the child point in the dfs at the time

        int* vis = (int*) calloc(M+1, sizeof(int));
        int* find = (int*) calloc(M+1, sizeof(int));
        int* position_node = (int*) calloc(M+1, sizeof(int));
        
        for (int j=0; j<M+1; j++) {
            vis[j] = 0;
            find[j] = j;
            position_node[j] = 1;
        }
        stk.push(largest_volume_point); //choose largest-volume point as root point
        vis[largest_volume_point] = 1;

        //cout << i << " .x ----- \n";
        //cout << node1 << "   " << node2 << '\n';

        //stop the search when the vertexes of edge have been found
        while (vis[node1]==0 || vis[node2]==0) {
            int t = stk.top();
            // cout << " sz : " << LG_vec[stk.top()].size() << '\n'; 
            for (int j=0; j<LG_vec[t].size(); j++) {
                int to = LG_vec[t][j].u;
                if (vis[to] != 1) {
                    position_node[stk.top()] = to;
                    vis[to] = 1;
                    stk.push(to);
                    break;
                }
                if (j == LG_vec[t].size() - 1) {
                    stk.pop();
                    find[t] = stk.top();
                    break;
                }
            }
        }

        //cout << i << " .xx ------ \n";

        //get the first point we found in dfs
        int node = stk.top();
        if (node == node2) node = node1;
        else node = node2; 

        //attain the no-weight distance between the first point that we found and LCA
        int d1 = 0;
        while (true) {
            node = find[node];
            d1++;
            if (find[node] == node) break;
        }

        // node is LCA
        re_dist = mst_dist[u] + mst_dist[v] - 2*mst_dist[node];
        off_tree_edge[i].re_dist = off_tree_edge[i].w * re_dist;

        free(vis);
        free(find);
        free(position_node);
    }


    
    gettimeofday(&end, NULL);
    printf("7 Cal Re(p): \t\t %f ms\n", (end.tv_sec-start1.tv_sec)*1000+(end.tv_usec-start1.tv_usec)/1000.0);
    //printf("%f\n", (end.tv_sec-start1.tv_sec)*1000+(end.tv_usec-start1.tv_usec)/1000.0);
    gettimeofday(&start1, NULL);
    


    //8. sort by effect resistance
    stable_sort(off_tree_edge, off_tree_edge+off_tree_cnt, compare_re);


    gettimeofday(&end, NULL);
    printf("8 Sort: \t\t %f ms\n", (end.tv_sec-start1.tv_sec)*1000+(end.tv_usec-start1.tv_usec)/1000.0);
    //printf("%f\n", (end.tv_sec-start1.tv_sec)*1000+(end.tv_usec-start1.tv_usec)/1000.0);
    gettimeofday(&start1, NULL);



    //9. add some edge into spanning tree
    int num_additive_tree = 0;
    bool* similar_edge = (bool*) calloc(off_tree_cnt, sizeof(bool));
    for (int i=0; i<=off_tree_cnt; i++) similar_edge[i] = false;

    int max_num_edges = max(off_tree_cnt/25, 2);
    int kf = 4;
    int Threads = omp_get_max_threads();
    int block_size = off_tree_cnt / Threads;

    //fprintf(out1, "threds %d blk %d\n", Threads, block_size);

    bool flag = false;
    
    for (int t=0; t<off_tree_cnt; t+=block_size){
        int min_bound = min(t+block_size, off_tree_cnt);

        vector<vector<int>> similarity_index;
        similarity_index.resize(block_size);

        //fprintf(out1, "\n range: %d %d \n", t, min_bound);    

        #pragma omp parallel for
        for(int i = t; i < min_bound; i++){
            int u = off_tree_edge[i].u;
            int v = off_tree_edge[i].v;

            if(!similar_edge[i]){
                int node1, node2;

                if (mst_root[u] == mst_root[v]) node1 = u, node2 = v;
                else {
                    node1 = (mst_root[u] != largest_volume_point) ? mst_root[u] : u;
                    node2 = (mst_root[v] != largest_volume_point) ? mst_root[v] : v;
                }
            
                // run targin to find dist
                stack<int> stk; //to show the process of dfs
                // int vis[M+1];  //to show whether a point has been gone through
                // int find[M+1];  //Joint search set
                // int position_node[M+1]; //restore the child point in the dfs at the time

                int* vis = (int*) calloc(M+1, sizeof(int));
                int* find = (int*) calloc(M+1, sizeof(int));
                int* position_node = (int*) calloc(M+1, sizeof(int));
                for (int j=0; j<M+1; j++) {
                    vis[j] = 0;
                    find[j] = j;
                    position_node[j] = 1;
                }
                stk.push(largest_volume_point); //choose largest-volume point as root point
                vis[largest_volume_point] = 1;


                //stop the search when the vertexes of edge have been found
                while (vis[node1]==0 || vis[node2]==0) {
                    int t = stk.top();
                    // cout << " sz : " << LG_vec[stk.top()].size() << '\n'; 
                    for (int j=0; j<LG_vec[t].size(); j++) {
                        int to = LG_vec[t][j].u;
                        if (vis[to] != 1) {
                            position_node[stk.top()] = to;
                            vis[to] = 1;
                            stk.push(to);
                            break;
                        }
                        if (j == LG_vec[t].size() - 1) {
                            stk.pop();
                            find[t] = stk.top();
                            break;
                        }
                    }
                }

                //get the first point we found in dfs
                int node = stk.top();
                if (node == node2) node = node1;
                else node = node2; 

                //attain the no-weight distance between the first point that we found and LCA
                int d1 = 0;
                while (true) {
                    node = find[node];
                    d1++;
                    if (find[node] == node) break;
                }
                
         
                int belta = min(mst_depth[u], mst_depth[v]) - mst_depth[node];

                bool mark1[M+1], mark2[M+1];

                belta_bfs(belta, mark1, u, M);
                belta_bfs(belta, mark2, v, M);


                for (int z=i+1; z<off_tree_cnt; z++) {
                    if(((mark1[off_tree_edge[z].u] && mark2[off_tree_edge[z].v]) 
                    || (mark1[off_tree_edge[z].v] && mark2[off_tree_edge[z].u]))) 
                        similarity_index[i-t].push_back(z);     
                }

                free(vis);
                free(find);
                free(position_node);  

            }
        }


        for(int i = t; i < min_bound; i++){
            if (num_additive_tree == max_num_edges) {
                flag = true;
                break;
            }
            if(similar_edge[i] == 0){
                num_additive_tree ++;
                // similarity_tree[i] = 1;
                spanning_tree[sp_tree_cnt++] = off_tree_edge[i];
                for(int l = 0; l < similarity_index[i-t].size(); l++){
                    int z = similarity_index[i-t][l];
                    similar_edge[z] = 1;
                }
            }
        }
        if(flag) break;
    }


    gettimeofday(&end, NULL);
    printf("9 Get similar edges: \t %f ms\n", (end.tv_sec-start1.tv_sec)*1000+(end.tv_usec-start1.tv_usec)/1000.0);
    //printf("%f\n", (end.tv_sec-start1.tv_sec)*1000+(end.tv_usec-start1.tv_usec)/1000.0);
    gettimeofday(&start1, NULL);


    gettimeofday(&end, NULL);
    //printf("Using time : %f ms\n", (end.tv_sec-start.tv_sec)*1000+(end.tv_usec-start.tv_usec)/1000.0);
    printf("%f\n", (end.tv_sec-start.tv_sec)*1000+(end.tv_usec-start.tv_usec)/1000.0);
    /**************************************************/
    /******************* End timing *******************/
    /**************************************************/

    FILE* out = fopen("result.txt", "w");
    for(int i=0; i<sp_tree_cnt; i++) fprintf(out, "%d %d\n", int(spanning_tree[i].u), int(spanning_tree[i].v));
    fclose(out);

    //fclose(out1);

    return 0;
}



