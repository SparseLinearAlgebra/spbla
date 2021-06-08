#include <cstdint>
#include "coo_utils.hpp"
#include "libutils/fast_random.h"
#include <vector>
#include <iostream>
#include <algorithm>


namespace clbool::coo_utils {

    using cpu_buffer = std::vector<uint32_t>;

    std::pair<matrix_dcsr_cpu, matrix_dcsr_cpu> generate_random_matrices_large(uint32_t max_size, uint32_t seed) {
        // попытаемся нагенерить штук 50 рядов по 32 - 63 элемента
        // создание второй матрицы
        matrix_dcsr_cpu a;
        matrix_dcsr_cpu b;
        {
            uint32_t random_to_fit = 150;
            uint32_t min_row_size = 129;
            cpu_buffer cols_indices;
            cpu_buffer rows_pointers;
            cpu_buffer rows_compressed;
            uint32_t rows = 50;

            if (max_size < rows) {
                throw std::runtime_error("too small matrix for generator");
            }

            FastRandom r(seed);
            rows_pointers.push_back(0);
            uint32_t nnz = 0;
            for (uint32_t i = 0; i < rows; ++i) {
                rows_compressed.push_back(i); // ох ну и пусть
                cpu_buffer curr_row(random_to_fit);
                for (uint32_t j = 0; j < random_to_fit; ++j) {
                    curr_row[j] = r.next(0, max_size);
                }
                std::sort(curr_row.begin(), curr_row.end());
                curr_row.erase(std::unique(curr_row.begin(), curr_row.end()), curr_row.end());
                uint32_t row_size = curr_row.size();
                if (row_size < min_row_size) continue;
                nnz += row_size;
                rows_pointers.push_back(nnz);
                cols_indices.insert(cols_indices.end(), curr_row.begin(), curr_row.end());
            }
            b = matrix_dcsr_cpu(rows_pointers, rows_compressed, cols_indices);
        }

        {
            cpu_buffer cols_indices{
                    1, 2, 7, 22, 23, 25, 28, 30,
                    10, 12, 13, 42,
                    4,
                    10, 15, 16, 23,
                    1, 5, 7, 14,
                    0, 2, 11, 14,
                    22, 24, 28, 30, 31, 32, 33, 34
            };

            cpu_buffer rows_pointers{
                    0, 8, 12, 13, 17, 21, 25, 33
            };

            cpu_buffer rows_compressed{
                    5, 33, 44, 234, 246, 567, 569
            };
            a = matrix_dcsr_cpu(rows_pointers, rows_compressed, cols_indices);
        }
        return std::make_pair(a, b);
    }

    std::pair<matrix_dcsr_cpu, matrix_dcsr_cpu> generate_random_matrices_esc(uint32_t max_size, uint32_t seed) {
        // попытаемся нагенерить штук 50 рядов по 32 - 63 элемента
        // создание второй матрицы
        matrix_dcsr_cpu a;
        matrix_dcsr_cpu b;
        {
            uint32_t random_to_fit = 64;
            uint32_t min_row_size = 33;
            cpu_buffer cols_indices;
            cpu_buffer rows_pointers;
            cpu_buffer rows_compressed;
            uint32_t rows = 50;

            if (max_size < rows) {
                throw std::runtime_error("too small matrix for generator");
            }

            FastRandom r(seed);
            rows_pointers.push_back(0);
            uint32_t nnz = 0;
            for (uint32_t i = 0; i < rows; ++i) {
                rows_compressed.push_back(i); // ох ну и пусть
                cpu_buffer curr_row(random_to_fit);
                for (uint32_t j = 0; j < random_to_fit; ++j) {
                    curr_row[j] = r.next(0, max_size);
                }
                std::sort(curr_row.begin(), curr_row.end());
                curr_row.erase(std::unique(curr_row.begin(), curr_row.end()), curr_row.end());
                uint32_t row_size = curr_row.size();
                if (row_size < min_row_size) continue;
                nnz += row_size;
                rows_pointers.push_back(nnz);
                cols_indices.insert(cols_indices.end(), curr_row.begin(), curr_row.end());
            }
            b = matrix_dcsr_cpu(rows_pointers, rows_compressed, cols_indices);
        }

        {

            cpu_buffer cols_indices{
                    1,
                    10, 12,
                    10, 15,
                    1, 5, 7, 14,
                    0, 2, 11, 14,
                    22, 24, 28, 30, 31, 32, 33, 34
            };

            cpu_buffer rows_pointers{
                    0, 1, 3, 5, 9, 13,
            };

            cpu_buffer rows_compressed{
                    5, 25, 26, 30, 31, 33
            };
            a = matrix_dcsr_cpu(rows_pointers, rows_compressed, cols_indices);
        }
        return std::make_pair(a, b);

    }


    void fill_random_matrix(cpu_buffer &rows, cpu_buffer &cols, uint32_t max_size) {
        if (max_size == 0) {
            if (!rows.empty() || !cols.empty()) {
                throw std::runtime_error("Incorrect generation");
            }
            return;
        }
        if (max_size < 1) {
            throw std::runtime_error("Incorrect generation");
        }
        uint32_t nnz = rows.size();
        FastRandom r(nnz);
        for (uint32_t i = 0; i < nnz; ++i) {
            rows[i] = r.next(0, max_size - 1);
            cols[i] = r.next(0, max_size - 1);
        }
    }

    void fill_random_matrix_pairs(matrix_coo_cpu_pairs &pairs, uint32_t max_size) {
        uint32_t n = pairs.size();
        FastRandom r(n);
        for (uint32_t i = 0; i < n; ++i) {
            pairs[i].first = r.next(0, max_size);
            pairs[i].second = r.next(0, max_size);
        }
    }

    void
    form_cpu_matrix(matrix_coo_cpu_pairs &matrix_out, const cpu_buffer &rows, const cpu_buffer &cols) {
        matrix_out.resize(rows.size());
        for (uint32_t i = 0; i < rows.size(); ++i) {
            matrix_out[i] = {rows[i], cols[i]};
        }
    }

    void get_vectors_from_cpu_matrix(cpu_buffer &rows_out, cpu_buffer &cols_out,
                                     const matrix_coo_cpu_pairs &matrix) {
        if (matrix.empty()) return;

        uint32_t n = matrix.size();

        rows_out.resize(matrix.size());
        cols_out.resize(matrix.size());

        for (uint32_t i = 0; i < n; ++i) {
            rows_out[i] = matrix[i].first;
            cols_out[i] = matrix[i].second;
        }

    }


    void check_correctness(const cpu_buffer &rows, const cpu_buffer &cols) {
        uint32_t n = rows.size();
        for (uint32_t i = 1; i < n; ++i) {
            if (rows[i] < rows[i - 1] || (rows[i] == rows[i - 1] && cols[i] < cols[i - 1])) {
                uint32_t start = i < 10 ? 0 : i - 10;
                uint32_t stop = i >= n - 10 ? n : i + 10;
                for (uint32_t k = start; k < stop; ++k) {
                    //TODO: all type of streams as parameter!!!!!!!!!
                    std::cout << k << ": (" << rows[k] << ", " << cols[k] << "), ";
                }
                std::cout << std::endl;
                throw std::runtime_error("incorrect result!");
            }
        }
        std::cout << "check finished, probably correct\n";
    }

    matrix_coo_cpu_pairs generate_coo_pairs_cpu(uint32_t pseudo_nnz, uint32_t max_size) {

        cpu_buffer rows(pseudo_nnz);
        cpu_buffer cols(pseudo_nnz);

        fill_random_matrix(rows, cols, max_size);

        matrix_coo_cpu_pairs m_cpu;
        form_cpu_matrix(m_cpu, rows, cols);
        std::sort(m_cpu.begin(), m_cpu.end());

        m_cpu.erase(std::unique(m_cpu.begin(), m_cpu.end()), m_cpu.end());

        return m_cpu;
    }

    matrix_coo_cpu generate_coo_cpu(uint32_t pseudo_nnz, uint32_t max_size) {

        matrix_coo_cpu_pairs m_pairs_cpu(pseudo_nnz);

        fill_random_matrix_pairs(m_pairs_cpu, max_size);

        std::sort(m_pairs_cpu.begin(), m_pairs_cpu.end());

        m_pairs_cpu.erase(std::unique(m_pairs_cpu.begin(), m_pairs_cpu.end()), m_pairs_cpu.end());

        cpu_buffer rows(m_pairs_cpu.size());
        cpu_buffer cols(m_pairs_cpu.size());

        for (size_t i = 0; i < m_pairs_cpu.size(); ++i) {
            rows[i] = m_pairs_cpu[i].first;
            cols[i] = m_pairs_cpu[i].second;
        }

        return matrix_coo_cpu(rows, cols);
    }

    matrix_dcsr_cpu coo_pairs_to_dcsr_cpu(const matrix_coo_cpu_pairs &matrix_coo) {
        if (matrix_coo.empty()) {
            return matrix_dcsr_cpu{};
        }

        cpu_buffer rows_pointers;
        cpu_buffer rows_compressed;
        cpu_buffer cols_indices;

        size_t position = 0;
        uint32_t curr_row = matrix_coo.front().first;
        rows_compressed.push_back(curr_row);
        rows_pointers.push_back(position);

        for (const auto &item: matrix_coo) {
            cols_indices.push_back(item.second);
            if (item.first != curr_row) {
                curr_row = item.first;
                rows_compressed.push_back(curr_row);
                rows_pointers.push_back(position);
            }
            position++;
        }
        rows_pointers.push_back(position);
        return matrix_dcsr_cpu(rows_pointers, rows_compressed, cols_indices);
    }

    matrix_dcsr_cpu coo_to_dcsr_cpu(const matrix_coo_cpu &matrix_coo) {

        if (matrix_coo.rows().empty()) {
            return matrix_dcsr_cpu();
        }

        cpu_buffer rows_pointers;
        cpu_buffer rows_compressed;
        const cpu_buffer& cols_indices(matrix_coo.cols());

        size_t position = 0;
        uint32_t curr_row = matrix_coo.rows()[0];
        rows_compressed.push_back(curr_row);
        rows_pointers.push_back(position);

        for (unsigned int i : matrix_coo.rows()) {
            if (i != curr_row) {
                curr_row = i;
                rows_compressed.push_back(curr_row);
                rows_pointers.push_back(position);
            }
            position++;
        }
        rows_pointers.push_back(position);

        return matrix_dcsr_cpu(rows_pointers, rows_compressed, cols_indices);
    }


    matrix_coo matrix_coo_from_cpu(Controls &controls, const matrix_coo_cpu_pairs &m_cpu,
                                   uint32_t nrows, uint32_t ncols) {
        if (m_cpu.empty()) {
            return matrix_coo(nrows, ncols);
        }

        cpu_buffer rows;
        cpu_buffer cols;

        get_vectors_from_cpu_matrix(rows, cols, m_cpu);

        uint32_t n_rows = nrows == -1 ? *std::max_element(rows.begin(), rows.end()) : nrows;
        uint32_t n_cols = ncols ==  -1 ? *std::max_element(cols.begin(), cols.end()) : ncols;
        uint32_t nnz = m_cpu.size();

        return matrix_coo(controls, rows.data(), cols.data(), n_rows, n_cols, nnz, true, true);
    }

    void
    matrix_addition_cpu(matrix_coo_cpu_pairs &matrix_out, const matrix_coo_cpu_pairs &matrix_a,
                        const matrix_coo_cpu_pairs &matrix_b) {

        std::merge(matrix_a.begin(), matrix_a.end(), matrix_b.begin(), matrix_b.end(),
                   std::back_inserter(matrix_out));

        matrix_out.erase(std::unique(matrix_out.begin(), matrix_out.end()), matrix_out.end());

    }

    void
    kronecker_product_cpu(matrix_coo_cpu_pairs &matrix_out, const matrix_coo_cpu_pairs &matrix_a,
                          const matrix_coo_cpu_pairs &matrix_b, uint32_t b_nrows, uint32_t b_ncols) {

        matrix_out.resize(matrix_a.size() * matrix_b.size());

        uint32_t i = 0;
        for (const auto &coord_a: matrix_a) {
            for (const auto &coord_b: matrix_b) {
                matrix_out[i] = coordinates(coord_a.first * b_nrows + coord_b.first,
                                            coord_a.second * b_ncols + coord_b.second);
                ++i;
            }
        }

        std::sort(matrix_out.begin(), matrix_out.end());
    }


    void print_matrix(const matrix_coo_cpu_pairs &m_cpu) {
        if (m_cpu.empty()) {
            std::cout << "empty matrix" << std::endl;
            return;
        }

        uint32_t curr_row = m_cpu.front().first;
        std::cout << "row " << curr_row << ": ";
        for (const auto &item: m_cpu) {
            if (item.first != curr_row) {
                curr_row = item.first;
                std::cout << std::endl;
                std::cout << "row " << curr_row << ": ";
            }
            std::cout << item.second << ", ";
        }
        std::cout << std::endl;
    }


    void print_matrix(const matrix_dcsr_cpu &m_cpu, uint32_t index) {

        if (m_cpu.cols().empty()) {
            std::cout << "empty matrix" << std::endl;
            return;
        }

        auto printNthRow = [&m_cpu](uint32_t i) {
            std::cout << i << ") row " << m_cpu.rows()[i] << ": ";
            uint32_t start = m_cpu.rpt()[i];
            uint32_t end = m_cpu.rpt()[i + 1];

            for (uint32_t j = start; j < end; ++j) {
                std::cout << "( " << j - start << ", " << m_cpu.cols()[j] << "), ";
            }
            std::cout << std::endl;
        };


        if (index != -1) {
            if (m_cpu.rows().size() - 1 < index) {
                std::cout << "cannot print row at pos " << index << ", matrix has " << m_cpu.rows().size() - 1
                          << " max index pos " << std::endl;
                return;
            }
            printNthRow(index);
            return;
        }

        uint32_t m_cpu_nzr = m_cpu.rows().size();

        for (uint32_t i = 0; i < m_cpu_nzr; ++i) {
            printNthRow(i);
        }

        std::cout << std::endl;
    }

    void print_matrix(Controls &controls, const matrix_dcsr &m_gpu, uint32_t index) {
        cpu_buffer rows_pointers(m_gpu.nzr() + 1);
        cpu_buffer rows_compressed(m_gpu.nzr());
        cpu_buffer cols_indices(m_gpu.nnz());

        controls.queue.enqueueReadBuffer(m_gpu.rpt_gpu(), CL_TRUE, 0,
                                         sizeof(uint32_t) * rows_pointers.size(), rows_pointers.data());
        controls.queue.enqueueReadBuffer(m_gpu.rows_gpu(), CL_TRUE, 0,
                                         sizeof(uint32_t) * rows_compressed.size(), rows_compressed.data());
        controls.queue.enqueueReadBuffer(m_gpu.cols_gpu(), CL_TRUE, 0,
                                         sizeof(uint32_t) * cols_indices.size(), cols_indices.data());
        print_matrix(matrix_dcsr_cpu(rows_pointers, rows_compressed, cols_indices), index);
    }

    void get_rows_pointers_and_compressed(cpu_buffer &rows_pointers,
                                          cpu_buffer &rows_compressed,
                                          const matrix_coo_cpu_pairs &matrix_cpu) {
        if (matrix_cpu.empty()) return;

        size_t position = 0;
        uint32_t curr_row = matrix_cpu.front().first;
        rows_compressed.push_back(curr_row);
        rows_pointers.push_back(position);
        for (const auto &item: matrix_cpu) {
            if (item.first != curr_row) {
                curr_row = item.first;
                rows_compressed.push_back(curr_row);
                rows_pointers.push_back(position);
            }
            position++;
        }
        rows_pointers.push_back(position);
        std::cout << std::endl;

    }

    void get_workload(cpu_buffer &workload,
                      const matrix_dcsr_cpu &a,
                      const matrix_dcsr_cpu &b
    ) {

        for (uint32_t i = 0; i < a.rows().size(); ++i) {
            workload[i] = 0;
            uint32_t start = a.rpt()[i];
            uint32_t end = a.rpt()[i + 1];
            for (uint32_t j = start; j < end; ++j) {
                auto it = std::find(b.rows().begin(), b.rows().end(), a.cols()[j]);
                if (it != b.rows().end()) {
                    uint32_t pos = it - b.rows().begin();
                    workload[i] += b.rpt()[pos + 1] - b.rpt()[pos];
                }
            }
        }
    }




//    void matrix_coo_to_dcsr_cpu(matrix_dcsr_cpu &out, const matrix_coo_cpu_pairs &in) {
//
//    }


    /*
     * штука нужна для сравнения с gpu, поэтому возвращать будем своего рода matrix_dcsr_cpu
     */

    void matrix_multiplication_cpu(matrix_dcsr_cpu &c,
                                   const matrix_dcsr_cpu &a,
                                   const matrix_dcsr_cpu &b) {

        uint32_t a_nzr = a.rows().size();
        cpu_buffer c_cols_indices;
        cpu_buffer c_rows_pointers;
        cpu_buffer c_rows_compressed;

        uint32_t current_pointer = 0;
        cpu_buffer current_row;
        for (uint32_t i = 0; i < a_nzr; ++i) {
            uint32_t start = a.rpt()[i];
            uint32_t end = a.rpt()[i + 1];
            bool is_row = false;

            for (uint32_t j = start; j < end; ++j) {
                auto it = std::find(b.rows().begin(), b.rows().end(), a.cols()[j]);
                if (it != b.rows().end()) {
                    if (!is_row) {
                        c_rows_pointers.push_back(current_pointer);
                        c_rows_compressed.push_back(a.rows()[i]);
                        is_row = true;
                    }
                    uint32_t pos = it - b.rows().begin();
                    uint32_t b_start = b.rpt()[pos];
                    uint32_t b_end = b.rpt()[pos + 1];
                    for (uint32_t k = b_start; k < b_end; ++k) {
                        current_row.push_back(b.cols()[k]);
                    }
                }
            }
            if (current_row.empty()) continue;

            std::sort(current_row.begin(), current_row.end());
            current_row.erase(std::unique(current_row.begin(), current_row.end()), current_row.end());
            c_cols_indices.insert(c_cols_indices.end(), current_row.begin(), current_row.end());

            current_pointer += current_row.size();
            current_row = cpu_buffer();
        }

        c_rows_pointers.push_back(current_pointer);
        c = matrix_dcsr_cpu(std::move(c_rows_pointers), std::move(c_rows_compressed), std::move(c_cols_indices));
    }

}