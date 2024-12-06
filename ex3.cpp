#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <omp.h>
#include <mpi.h>
#include <algorithm>  // Adiciona a biblioteca <algorithm>
#include <filesystem> // Adiciona para trabalhar com diretórios (C++17)

namespace fs = std::filesystem;

// Função para transcrever uma sequência de DNA para RNA
void transcreverDNAparaRNA(const std::string& sequencia_dna, std::string& sequencia_rna) {
    #pragma omp parallel for
    for (size_t i = 0; i < sequencia_dna.length(); ++i) {
        switch (toupper(sequencia_dna[i])) {
            case 'A':
                sequencia_rna[i] = 'U';
                break;
            case 'T':
                sequencia_rna[i] = 'A';
                break;
            case 'C':
                sequencia_rna[i] = 'G';
                break;
            case 'G':
                sequencia_rna[i] = 'C';
                break;
            default:
                sequencia_rna[i] = 'N'; // Base desconhecida
                break;
        }
    }
}

// Função para contar o número de códons AUG em uma sequência de RNA
int contarCódonsAUG(const std::string& sequencia_rna) {
    int contagem = 0;
    #pragma omp parallel for reduction(+:contagem)
    for (size_t i = 0; i < sequencia_rna.length() - 2; ++i) {
        if (sequencia_rna.substr(i, 3) == "AUG") {
            contagem++;
        }
    }
    return contagem;
}

int main(int argc, char* argv[]) {
    int rank, total_processos;
    MPI_Init(&argc, &argv); // Inicializa o MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtém o rank do processo
    MPI_Comm_size(MPI_COMM_WORLD, &total_processos); // Obtém o número total de processos

    // Diretório onde estão os arquivos .fa
    std::string caminho_diretorio = "."; // Pode ser alterado para o diretório desejado

    // Lista de arquivos .fa no diretório
    std::vector<std::string> arquivos_entrada;

    // Percorre o diretório e adiciona os arquivos .fa na lista
    for (const auto& entry : fs::directory_iterator(caminho_diretorio)) {
        if (entry.is_regular_file() && entry.path().extension() == ".fa") {
            arquivos_entrada.push_back(entry.path().string());
        }
    }

    // Laço para processar cada arquivo
    for (size_t arquivo_idx = 0; arquivo_idx < arquivos_entrada.size(); ++arquivo_idx) {
        std::string caminho_arquivo = arquivos_entrada[arquivo_idx]; // Arquivo atual a ser processado
        std::string sequencia_dna;

        if (rank == 0) {
            // Processo raiz lê o arquivo FASTA
            std::ifstream arquivo(caminho_arquivo);
            if (!arquivo.is_open()) {
                std::cerr << "Erro ao abrir o arquivo " << caminho_arquivo << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }

            std::string linha;
            bool eh_cabecalho = true;
            while (std::getline(arquivo, linha)) {
                if (linha.empty()) continue; // Ignora linhas vazias
                if (linha[0] == '>') {
                    eh_cabecalho = false; // Ignora a linha de cabeçalho
                    continue;
                }
                if (!eh_cabecalho) {
                    // Concatena a sequência de DNA, removendo espaços
                    linha.erase(remove_if(linha.begin(), linha.end(), ::isspace), linha.end());
                    sequencia_dna += linha;
                }
            }
            arquivo.close();
        }

        // Broadcast do tamanho da sequência de DNA para todos os processos
        size_t comprimento_total = sequencia_dna.length();
        MPI_Bcast(&comprimento_total, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

        // Cada processo aloca espaço para a sequência de DNA
        if (rank != 0) {
            sequencia_dna.resize(comprimento_total);
        }

        // Broadcast da sequência de DNA para todos os processos
        MPI_Bcast(const_cast<char*>(sequencia_dna.c_str()), comprimento_total, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Determina a divisão das sequências entre os processos
        size_t comprimento_por_processo = comprimento_total / total_processos;
        size_t inicio = rank * comprimento_por_processo;
        size_t fim = (rank == total_processos - 1) ? comprimento_total : (rank + 1) * comprimento_por_processo;
        size_t comprimento_local = fim - inicio;

        // Extrai a subsequência local para cada processo
        std::string sequencia_dna_local = sequencia_dna.substr(inicio, comprimento_local);
        std::string sequencia_rna_local(comprimento_local, ' ');

        // Transcreve DNA para RNA usando OpenMP
        transcreverDNAparaRNA(sequencia_dna_local, sequencia_rna_local);

        // Contagem dos códons AUG na subsequência de RNA local
        int contagem_local = contarCódonsAUG(sequencia_rna_local);

        // Realiza a redução para somar as contagens de todos os processos
        int contagem_total;
        MPI_Reduce(&contagem_local, &contagem_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // Processo raiz exibe a contagem total dos códons AUG
        if (rank == 0) {
            std::cout << "Total de códons AUG (início de proteínas) no arquivo " 
                      << arquivos_entrada[arquivo_idx] << ": " 
                      << contagem_total << std::endl;
        }
    }

    MPI_Finalize(); // Finaliza o MPI
    return 0;
}
