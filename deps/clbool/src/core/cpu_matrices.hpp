#pragma once

namespace clbool {

    using coordinates = std::pair<uint32_t, uint32_t>;
    using matrix_coo_cpu_pairs = std::vector<coordinates>;
    using cpu_buffer = std::vector<uint32_t>;

    class matrix_dcsr_cpu {
        cpu_buffer _rpt;
        cpu_buffer _rows;
        cpu_buffer _cols;

    public:
        matrix_dcsr_cpu(cpu_buffer rpt, cpu_buffer rows, cpu_buffer cols)
                : _rpt(std::move(rpt)), _rows(std::move(rows)),
                  _cols(std::move(cols)) {}

        matrix_dcsr_cpu() = default;

        matrix_dcsr_cpu &operator=(matrix_dcsr_cpu other) {
            _rpt = std::move(other._rpt);
            _rows = std::move(other._rows);
            _cols = std::move(other._cols);
            return *this;
        }

        cpu_buffer &rpt() {
            return _rpt;
        }

        cpu_buffer &rows() {
            return _rows;
        }

        cpu_buffer &cols() {
            return _cols;
        }

        const cpu_buffer &rpt() const {
            return _rpt;
        }

        const cpu_buffer &rows() const {
            return _rows;
        }

        const cpu_buffer &cols() const {
            return _cols;
        }

    };


    class matrix_coo_cpu {
        cpu_buffer _rows;
        cpu_buffer _cols;

    public:
        matrix_coo_cpu(cpu_buffer rows, cpu_buffer cols)
                : _rows(std::move(rows))
                , _cols(std::move(cols))
                {}

        matrix_coo_cpu() = default;

        matrix_coo_cpu &operator=(matrix_coo_cpu other) {
            _rows = std::move(other._rows);
            _cols = std::move(other._cols);
            return *this;
        }

        cpu_buffer &rows() {
            return _rows;
        }

        cpu_buffer &cols() {
            return _cols;
        }

        const cpu_buffer &rows() const {
            return _rows;
        }

        const cpu_buffer &cols() const {
            return _cols;
        }

        void transpose() {
            std::swap(_rows, _cols);
            std::vector<std::pair<uint32_t, uint32_t>> pairs;

            for (size_t i = 0; i < _rows.size(); ++i) {
                pairs.emplace_back(_rows[i], _cols[i]);
            }

            std::sort(pairs.begin(), pairs.end());

            for (size_t i = 0; i < _rows.size(); ++i) {
                _rows[i] = pairs[i].first;
                _cols[i] = pairs[i].second;
            }

        }

    };

    class matrix_csr_cpu {
        cpu_buffer _rpt;
        cpu_buffer _cols;

        uint32_t _nrows;
        uint32_t _ncols;

    public:
        matrix_csr_cpu(cpu_buffer rpt, cpu_buffer cols, uint32_t nrows, uint32_t ncols)
                : _rpt(std::move(rpt))
                ,_cols(std::move(cols))
                ,_nrows(nrows)
                ,_ncols(ncols)
                {}

        matrix_csr_cpu(uint32_t nrows, uint32_t ncols)
        : _nrows(nrows)
        , _ncols(ncols)
        {}

        matrix_csr_cpu() = default;

        matrix_csr_cpu &operator=(matrix_csr_cpu other) {
            _rpt = std::move(other._rpt);
            _cols = std::move(other._cols);
            return *this;
        }

        cpu_buffer &rpt() {
            return _rpt;
        }

        cpu_buffer &cols() {
            return _cols;
        }

        const cpu_buffer &rpt() const {
            return _rpt;
        }

        const cpu_buffer &cols() const {
            return _cols;
        }

        const uint32_t nrows() {
            return _nrows;
        }

        const uint32_t ncols() {
            return _ncols;
        }

        uint32_t nrows() const {
            return _nrows;
        }

        uint32_t ncols() const {
            return _ncols;
        }


    };
}