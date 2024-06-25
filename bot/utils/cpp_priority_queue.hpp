#include <functional>
#include <queue>
#include <utility>

using cpp_pq = std::priority_queue<std::pair<double,std::pair<int,int>>,std::vector<std::pair<double,std::pair<int,int>>>,std::function<bool(std::pair<double,std::pair<int,int>>,std::pair<double,std::pair<int,int>>)>>;