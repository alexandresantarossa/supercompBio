#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <filesystem>

const char BASE_A = 'a';
const char BASE_T = 't';
const char BASE_C = 'c';
const char BASE_G = 'g';

// Função para contar as bases em uma parte da sequência de DNA
void contarBases(const std::string& sequencia, int& conta_a, int& conta_t, int& conta_c, int& conta_g) {
    #pragma omp parallel for reduction(+:conta_a, conta_t, conta_c, conta_g)
    for (size_t i = 0; i < sequencia.size(); ++i) {
        char base = sequencia[i];
        if (base == BASE_A) ++conta_a;
        else if (base == BASE_T) ++conta_t;
        else if (base == BASE_C) ++conta_c;
        else if (base == BASE_G) ++conta_g;
    }
}

// Função para ler a sequência de DNA de um arquivo
std::string lerSequenciaDNA(const std::string& caminho) {
    std::ifstream arquivo(caminho);
    if (!arquivo.is_open()) {
        throw std::runtime_error("Erro ao abrir o arquivo " + caminho);
    }

    std::string linha, sequencia;
    while (std::getline(arquivo, linha)) {
        if (linha.empty() || linha[0] == '>') continue; // Ignora cabeçalhos
        sequencia += linha;
    }
    return sequencia;
}

int main(int argc, char* argv[]) {
    int rank, total_processos;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processos);

    // Diretório de arquivos FASTA (atualmente na mesma pasta do código)
    std::string directory = ".";  // Usando o diretório atual

    // Verificar se o diretório existe
    if (!std::filesystem::exists(directory)) {
        if (rank == 0) {
            std::cerr << "Erro: O diretório " << directory << " não existe.\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Usar a biblioteca std::filesystem para explorar os arquivos no diretório
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().extension() == ".fa") {  // Apenas arquivos com extensão .fa
            files.push_back(entry.path().string());
        }
    }

    int num_files = files.size();  // Número de arquivos encontrados

    // Garantir que há arquivos para processar
    if (num_files == 0) {
        if (rank == 0) {
            std::cerr << "Nenhum arquivo .fa encontrado no diretório.\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Dividir os arquivos entre os processos
    int files_per_process = num_files / total_processos;
    int start_index = rank * files_per_process;
    int end_index = (rank + 1) * files_per_process;

    // Se houver arquivos extras (não divididos igualmente), o último processo pega o restante
    if (rank == total_processos - 1) {
        end_index = num_files;
    }

    // Variáveis para contagem das bases
    int globais_a = 0, globais_t = 0, globais_c = 0, globais_g = 0;

    // Processar os arquivos atribuídos ao processo
    for (int i = start_index; i < end_index; ++i) {
        std::string nome_arquivo = files[i];  // Nome do arquivo

        std::string sequencia_completa;
        try {
            sequencia_completa = lerSequenciaDNA(nome_arquivo);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        // Contadores locais para as bases
        int locais_a = 0, locais_t = 0, locais_c = 0, locais_g = 0;

        // Contar as bases no arquivo atual
        contarBases(sequencia_completa, locais_a, locais_t, locais_c, locais_g);

        // Somar os resultados locais com os globais
        MPI_Reduce(&locais_a, &globais_a, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&locais_t, &globais_t, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&locais_c, &globais_c, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&locais_g, &globais_g, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // Exibir os resultados no rank 0
    if (rank == 0) {
        std::cout << "Contagem final de bases:\n";
        std::cout << "A: " << globais_a << "\n";
        std::cout << "T: " << globais_t << "\n";
        std::cout << "C: " << globais_c << "\n";
        std::cout << "G: " << globais_g << "\n";
    }

    MPI_Finalize();
    return 0;
}
