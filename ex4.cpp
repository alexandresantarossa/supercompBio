#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <omp.h>
#include <mpi.h>
#include <algorithm>
#include <filesystem>
#include <numeric>  // Para usar std::accumulate
#include <map>      // Adicionando o cabeçalho necessário para usar std::map

namespace fs = std::filesystem;

// Função para transcrever DNA para RNA
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

// Função para traduzir a sequência de RNA em proteína
void traduzirRNAParaProteina(const std::string& sequencia_rna, std::vector<int>& proteina, bool& encontrou_stop) {
    for (size_t i = 0; i < sequencia_rna.length() - 2; i += 3) {
        std::string codon = sequencia_rna.substr(i, 3);
        
        if (codon == "AUG") {
            proteina.push_back(1);  // Metionina (início)
        } else if (codon == "UGA") {
            encontrou_stop = true;
            break;  // Códon STOP
        } else if (codon == "CGA" || codon == "CCG" || codon == "CCU" || codon == "CCC") {
            proteina.push_back(2);  // Prolina
        } else if (codon == "UCU" || codon == "UCA" || codon == "UCG" || codon == "UCC") {
            proteina.push_back(3);  // Serina
        } else if (codon == "CAG" || codon == "CAA") {
            proteina.push_back(4);  // Glutamina
        } else if (codon == "ACA" || codon == "ACC" || codon == "ACU" || codon == "ACG") {
            proteina.push_back(5);  // Treonina
        } else if (codon == "UGC" || codon == "UGU") {
            proteina.push_back(6);  // Cisteína
        } else if (codon == "GUG" || codon == "GUA" || codon == "GUC" || codon == "GUU") {
            proteina.push_back(7);  // Valina
        }
    }
}

// Função principal
int main(int argc, char* argv[]) {
    int rank, total_processos;
    MPI_Init(&argc, &argv);  // Inicializa o MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Obtém o rank do processo
    MPI_Comm_size(MPI_COMM_WORLD, &total_processos);  // Obtém o número total de processos

    std::string caminho_diretorio = ".";  // Diretório onde estão os arquivos .fa
    std::vector<std::string> arquivos_entrada;

    // Percorre o diretório e adiciona os arquivos .fa na lista
    for (const auto& entry : fs::directory_iterator(caminho_diretorio)) {
        if (entry.is_regular_file() && entry.path().extension() == ".fa") {
            arquivos_entrada.push_back(entry.path().string());
        }
    }

    // Laço para processar cada arquivo .fa
    for (size_t arquivo_idx = 0; arquivo_idx < arquivos_entrada.size(); ++arquivo_idx) {
        std::string caminho_arquivo = arquivos_entrada[arquivo_idx];  // Arquivo atual a ser processado
        std::string sequencia_dna, sequencia_rna;

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
                if (linha.empty()) continue;  // Ignora linhas vazias
                if (linha[0] == '>') {
                    eh_cabecalho = false;  // Ignora a linha de cabeçalho
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

        // Transcrição do DNA para RNA
        sequencia_rna.resize(sequencia_dna.length());
        transcreverDNAparaRNA(sequencia_dna, sequencia_rna);

        // Broadcast do tamanho da sequência de RNA para todos os processos
        size_t comprimento_total = sequencia_rna.length();
        MPI_Bcast(&comprimento_total, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

        // Cada processo aloca espaço para a sequência de RNA
        if (rank != 0) {
            sequencia_rna.resize(comprimento_total);
        }

        // Broadcast da sequência de RNA para todos os processos
        MPI_Bcast(const_cast<char*>(sequencia_rna.c_str()), comprimento_total, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Determina a divisão das sequências entre os processos
        size_t comprimento_por_processo = comprimento_total / total_processos;
        size_t inicio = rank * comprimento_por_processo;
        size_t fim = (rank == total_processos - 1) ? comprimento_total : (rank + 1) * comprimento_por_processo;

        // Extrai a subsequência local para cada processo
        std::string sequencia_rna_local = sequencia_rna.substr(inicio, fim - inicio);
        std::vector<int> proteina_local;
        bool encontrou_stop = false;  // Variável para controle do códon STOP

        // Traduz a subsequência de RNA para proteína
        traduzirRNAParaProteina(sequencia_rna_local, proteina_local, encontrou_stop);

        // Calcular o número de elementos enviados por cada processo
        int tamanho_local = proteina_local.size();
        std::vector<int> tamanhos_processos(total_processos, 0);

        // Determina o número total de elementos a ser enviado
        MPI_Gather(&tamanho_local, 1, MPI_INT, tamanhos_processos.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Agora, o processo mestre (rank 0) pode alocar o espaço correto para o vetor global
        int tamanho_total = 0;
        if (rank == 0) {
            tamanho_total = std::accumulate(tamanhos_processos.begin(), tamanhos_processos.end(), 0);
        }

        std::vector<int> proteina_global(tamanho_total);

        // Usar MPI_Gatherv para reunir as proteínas de todos os processos
        std::vector<int> deslocamentos(total_processos, 0);
        if (rank == 0) {
            for (int i = 1; i < total_processos; ++i) {
                deslocamentos[i] = deslocamentos[i - 1] + tamanhos_processos[i - 1];
            }
        }

        MPI_Gatherv(proteina_local.data(), tamanho_local, MPI_INT,
                    proteina_global.data(), tamanhos_processos.data(),
                    deslocamentos.data(), MPI_INT, 0, MPI_COMM_WORLD);

        // Mapa para os códons e seus significados
        std::map<int, std::string> codons = {
            {1, "AUG - Metionina (início)"},
            {2, "CGA, CCG, CCU, CCC - Prolina"},
            {3, "UCU, UCA, UCG, UCC - Serina"},
            {4, "CAG, CAA - Glutamina"},
            {5, "ACA, ACC, ACU, ACG - Treonina"},
            {6, "UGC, UGU - Cisteína"},
            {7, "GUG, GUA, GUC, GUU - Valina"}
        };

        // Processo raiz monta a proteína completa e salva o resultado
        if (rank == 0) {
            std::ofstream arquivo_saida("proteina_traduzida" + std::to_string(arquivo_idx + 1) + ".txt");
            if (!arquivo_saida.is_open()) {
                std::cerr << "Erro ao salvar o arquivo da proteína." << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }

            // Imprime a legenda
            arquivo_saida << "Legenda dos códons:\n";
            for (const auto& [numero, descricao] : codons) {
                arquivo_saida << numero << " - " << descricao << "\n";
            }

            // Imprime a proteína traduzida
            arquivo_saida << "\nProteína traduzida:\n";
            for (const auto& aminoacido : proteina_global) {
                if (aminoacido != 0) arquivo_saida << aminoacido << " ";
            }
            arquivo_saida.close();
            std::cout << "Proteína traduzida salva em 'proteina_traduzida" << arquivo_idx + 1 << ".txt'." << std::endl;
        }
    }

    MPI_Finalize();  // Finaliza o MPI
    return 0;
}
