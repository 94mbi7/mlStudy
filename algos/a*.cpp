#include <bits/stdc++.h>
using namespace std;

queue<int> q;
int visited[6] = {0};
int heuristic[] = {5, 4, 2, 2, 1, 0};

vector<int> bfs(vector<vector<int>> &a) {
    vector<int> path;
    int sum = 0;

    q.push(0);
    visited[0] = 1;
    path.push_back(0);

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        int nextNode = -1;
        int minHeuristic = INT_MAX;

        for (int j = 0; j < 6; j++) {
            if (a[current][j] != 0 && !visited[j]) {
                if (heuristic[j] < minHeuristic) {
                    minHeuristic = heuristic[j];
                    nextNode = j;
                }
            }
        }

        if (nextNode != -1) {
            q.push(nextNode);
            visited[nextNode] = 1;
            path.push_back(nextNode);
            sum += a[current][nextNode]; 
        }
    }

    path.push_back(sum); 
    return path;
}

int main() {
    vector<vector<int>> adj = {
        {0, 2, 0, 1, 0, 0},
        {0, 0, 3, 2, 0, 0},
        {0, 0, 0, 0, 4, 0},
        {0, 0, 0, 0, 2, 3},
        {0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0}
    };

    vector<int> p = bfs(adj);

    cout << "Path: ";
    for (int i = 0; i < p.size() - 1; i++) {
        cout << p[i];
        if (i < p.size() - 2) cout << " -> ";
    }
    cout << "\nLength of path: " << p.back() << endl;

    return 0;
}
