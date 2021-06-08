#include <fstream>
#include <iostream>
#include <string>
#include <regex>


void replace_slash(std::ofstream &fout, const char *word) {
    for (int i = 0; word[i] != '\0'; ++i) {
        if (word[i] == '/') {
            fout << "___" ;
        } else {
            fout << word[i];
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <header_hile> <kernels_list>" << std::endl;
        return 1;
    }

    std::string headerFilename(argv[1]);
    std::ofstream fout(headerFilename);
    fout << "#include <unordered_map>" << std::endl;
    for (int i = 2; i < argc; ++i) {
        fout << "#include \"";
        replace_slash(fout, argv[i]);
        fout << ".h\"" << std::endl;
    }
    fout << "struct KernelSource {" << std::endl
         << "    const char* kernel;" << std::endl
         << "    size_t length;" << std::endl
         << "};" << std::endl;

    fout << "static const std::unordered_map<std::string, KernelSource> HeadersMap = {" << std::endl;

    std::regex slash("/");
    for (int i = 2; i < argc; ++i) {
           fout << "        {\"" <<  argv[i] << "\"" << ", {";
           replace_slash(fout, argv[i]);
           fout << "_kernel, ";
           replace_slash(fout, argv[i]);
           fout << "_kernel_length}}," << std::endl;
    }
    fout << "};" << std::endl;
    fout.close();

    return 0;
}
