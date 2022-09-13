//
// Created by vukasin on 8/16/21.
//

#ifndef MOOREPENROSE_FILE_UTILS_H
#define MOOREPENROSE_FILE_UTILS_H
#include <sys/stat.h>

inline bool file_exists(const std::string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

template<typename Time_type>
void write_results(unsigned int n,
                   unsigned int m,
                   const std::string& algo,
                   const std::string& processor,
                   const std::string& type,
                   Time_type time) {

    std::ofstream outfile;
    bool write_header = !file_exists("results.txt");
    outfile.open("results.txt", std::ios_base::app);
    outfile << std::left;

    if (write_header) {
        outfile << std::setw(12) << "n";
        outfile << std::setw(12) << "m";
        outfile << std::setw(12) << "type";
        outfile << std::setw(12) << "processor";
        outfile << std::setw(17) << "algorithm";
        outfile << std::setw(12) << "time (ms)";
        outfile << std::endl;
    }

    outfile << std::setw(12) << n;
    outfile << std::setw(12) << m;
    outfile << std::setw(12) << type;
    outfile << std::setw(12) << processor;
    outfile << std::setw(17) << algo;
    outfile << std::setw(12) << time;

    outfile << std::endl;

    outfile.close();
}

#endif //MOOREPENROSE_FILE_UTILS_H
