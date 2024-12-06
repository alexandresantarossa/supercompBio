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

        // Preparar para o MPI_Gatherv
        int quantidade_envio = comprimento_local;

        std::vector<int> quantidade_recebida;
        if (rank == 0) {
            quantidade_recebida.resize(total_processos);
        }
        MPI_Gather(&quantidade_envio, 1, MPI_INT, quantidade_recebida.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Processo raiz calcula os deslocamentos
        std::vector<int> deslocamentos;
        std::string sequencia_rna_global;
        if (rank == 0) {
            deslocamentos.resize(total_processos, 0);
            for (int i = 1; i < total_processos; ++i) {
                deslocamentos[i] = deslocamentos[i-1] + quantidade_recebida[i-1];
            }
            sequencia_rna_global.resize(comprimento_total, ' ');
        }

        // Preparar os buffers de envio
        MPI_Gatherv(sequencia_rna_local.data(), quantidade_envio, MPI_CHAR,
                    rank == 0 ? &sequencia_rna_global[0] : nullptr, 
                    rank == 0 ? quantidade_recebida.data() : nullptr,
                    rank == 0 ? deslocamentos.data() : nullptr,
                    MPI_CHAR, 0, MPI_COMM_WORLD);

        // Processo raiz salva o RNA transcrito em um arquivo
        if (rank == 0) {
            // Salva o RNA transcrito em um arquivo de saída com nome sequencial
            std::ofstream arquivo_saida("transcricao" + std::to_string(arquivo_idx + 1) + ".fa");
            if (!arquivo_saida.is_open()) {
                std::cerr << "Erro ao salvar o arquivo de RNA." << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            arquivo_saida << ">RNA_Transcrito_" << arquivo_idx + 1 << "\n" << sequencia_rna_global << std::endl;
            arquivo_saida.close();
            std::cout << "RNA transcrito salvo em 'transcricao" << arquivo_idx + 1 << ".fa'." << std::endl;
        }
    }

    MPI_Finalize(); // Finaliza o MPI
    return 0;
}
