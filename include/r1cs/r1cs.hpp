#pragma once
#include <cstddef>
#include <vector>
#include <map>

#include "r1cs/spmat.hpp"

namespace LightZK {

/*
    类型 S 需要支持:
    - 默认构造 S{0}
    - 构造/赋值 S{0}, S{1}
    - 算术 +, *
*/

enum class VariableType { Public, Private };

template<typename S> class LinearCombination;

#define HALF_MAX_SIZE_T ((size_t(-1) >> 1) + 1)

template<typename S>
class R1CSManager {
    size_t public_count = 1;
    size_t private_count = HALF_MAX_SIZE_T;
    
    struct Constraint { 
        LinearCombination<S> a, b, c; 
        Constraint(LinearCombination<S> a_, LinearCombination<S> b_, LinearCombination<S> c_) : a(a_), b(b_), c(c_) {}
    };
    std::vector<Constraint> constraints;
public:
    size_t num_constraints() const { return constraints.size(); }
    size_t num_variables() const { return public_count + (private_count - HALF_MAX_SIZE_T); }

    size_t get_id(VariableType type) {
        if (public_count >= HALF_MAX_SIZE_T || private_count < HALF_MAX_SIZE_T) throw std::runtime_error("R1CSManager: variable count overflow");
        switch (type) {
            case VariableType::Public: return public_count++;
            case VariableType::Private: return private_count++;
        }
    }

    void add_constraint(LinearCombination<S> out, LinearCombination<S> in) {
        constraints.push_back({in, LinearCombination<S>(S{1}), out});
    }

    void add_constraint(LinearCombination<S> out, LinearCombination<S> in1, LinearCombination<S> in2) {
        constraints.push_back({in1, in2, out});
    }

    void gen_spmat(SparseMatrix<S> &A, SparseMatrix<S> &B, SparseMatrix<S> &C, bool padding = true) {
        size_t rows = constraints.size();
        size_t cols = public_count + (private_count - HALF_MAX_SIZE_T);

        if (padding) {
            size_t padded_rows = 1, padded_cols = 1;
            while (padded_rows < rows) padded_rows <<= 1;
            while (padded_cols < cols) padded_cols <<= 1;
            rows = padded_rows, cols = padded_cols;
        }

        A.row_ptr.resize(rows + 1, 0);
        B.row_ptr.resize(rows + 1, 0);
        C.row_ptr.resize(rows + 1, 0);
        
        A.num_cols = cols;
        B.num_cols = cols;
        C.num_cols = cols;

        auto dense = [=] (size_t id) { return (id < HALF_MAX_SIZE_T) ? id : (id - HALF_MAX_SIZE_T + public_count); };

        for (size_t r = 0; r < constraints.size(); r++) {
            const auto& [a, b, c] = constraints[r];
            for (const auto& [id, coeff] : a.get_terms()) { A.col_idx.push_back(dense(id)); A.values.push_back(coeff); }
            for (const auto& [id, coeff] : b.get_terms()) { B.col_idx.push_back(dense(id)); B.values.push_back(coeff); }
            for (const auto& [id, coeff] : c.get_terms()) { C.col_idx.push_back(dense(id)); C.values.push_back(coeff); }
            A.row_ptr[r + 1] = A.col_idx.size();
            B.row_ptr[r + 1] = B.col_idx.size();
            C.row_ptr[r + 1] = C.col_idx.size();
        }

        for (size_t r = constraints.size(); r < rows; r++) {
            A.row_ptr[r + 1] = A.col_idx.size();
            B.row_ptr[r + 1] = B.col_idx.size();
            C.row_ptr[r + 1] = C.col_idx.size();
        }
    }
};

template<typename S>
class Variable {
    size_t id;
    VariableType type;
public:
    Variable(VariableType type_, R1CSManager<S>& mgr_) : type(type_), id(mgr_.get_id(type_)) {}
    operator LinearCombination<S>() const;

    size_t get_id() const { return id; }
};

template<typename S>
class LinearCombination {
    std::map<size_t, S> terms;
public:
    LinearCombination<S>() { terms[0] = S{0}; }
    LinearCombination<S>(const S& constant) { terms[0] = constant; }
    LinearCombination<S>(const Variable<S>& var) { terms[var.get_id()] = S{1}; }
    
    LinearCombination<S> operator+(const LinearCombination<S>& other) const {
        LinearCombination<S> result = *this;
        for (const auto& [id, coeff] : other.terms) result.terms[id] += coeff;
        return std::move(result);
    }
    LinearCombination<S>& operator+=(const Variable<S>& var) { terms[var.get_id()] += S{1}; return *this; }

    const std::map<size_t, S>& get_terms() const { return terms; }
};

template<typename S>
Variable<S>::operator LinearCombination<S>() const { return LinearCombination<S>(*this); }

/*
    == 理想使用方式 ==
    {
        R1CSManager<S> mgr;
        vector<Variable<S>> A, B, C, P;
        for (int i = 0; i < n * n; i++) A.push_back(Variable<S>(VariableType::Public, mgr));
        for (int i = 0; i < n * n; i++) B.push_back(Variable<S>(VariableType::Public, mgr));
        for (int i = 0; i < n * n; i++) C.push_back(Variable<S>(VariableType::Public, mgr));
        for (int i = 0; i < n * n * n; i++) P.push_back(Variable<S>(VariableType::Private, mgr));
        
        for (int i = 0; i < n; i++) 
            for (int j = 0; j < n; j++) 
                for (int k = 0; k < n; k++) 
                    mgr.add_constraint(P[i * n * n + j * n + k], A[i * n + k], B[k * n + j]);
        
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
            LinearCombination<S> sum;
            for (int k = 0; k < n; k++) sum += P[i * n * n + j * n + k];
            mgr.add_constraint(C[i * n + j], sum);
        }

        SparseMatrix<S> R1CS_A, R1CS_B, R1CS_C;
        mgr.get_r1cs(R1CS_A, R1CS_B, R1CS_C);
    }
*/

} // namespace LightZK