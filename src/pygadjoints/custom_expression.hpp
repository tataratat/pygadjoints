#pragma once
namespace gismo::expr {

/*
Expression for the Frobenius matrix product between a matrix of matrices and a
single matrix with cardinality 1
*/
template <typename E1, typename E2>
class frobenius_prod_expr : public _expr<frobenius_prod_expr<E1, E2>> {
public:
  typedef typename E1::Scalar Scalar;
  enum {
    ScalarValued = 0,
    Space = E1::Space,
    ColBlocks = 0 // E1::ColBlocks || E2::ColBlocks
  };

private:
  typename E1::Nested_t _u;
  typename E2::Nested_t _v;

  mutable gsMatrix<Scalar> res;

public:
  frobenius_prod_expr(_expr<E1> const &u, _expr<E2> const &v) : _u(u), _v(v) {
    // todo: add check() functions, which will evaluate expressions on an empty
    // matrix (no points) to setup initial dimensions ???
    GISMO_ASSERT(_u.rows() == _v.rows(), "Wrong dimensions "
                                             << _u.rows() << "!=" << _v.rows()
                                             << " in % operation");
    GISMO_ASSERT(_u.cols() == _v.cols(), "Wrong dimensions "
                                             << _u.cols() << "!=" << _v.cols()
                                             << " in %operation");
  }

  const gsMatrix<Scalar> &eval(const index_t k) const {
    // Evaluate Expressions and cardinality
    const index_t u_r = _u.rows();
    const index_t u_c = _u.cols();

    // Cardinality impl refers to the cols im a matrix
    auto A = _u.eval(k);
    auto B = _v.eval(k);
    const index_t A_rows = A.rows() / u_r;
    const index_t A_cols = A.cols() / u_c;
    GISMO_ASSERT(
        _v.cardinality() == 1,
        "Expression is only for second expressions with cardinality 1");
    GISMO_ASSERT((u_r == _v.cols()) && (u_c == _v.cols()),
                 "Both expressions need to be same size");
    GISMO_ASSERT(B.size() == _v.rows() * _v.cols(),
                 "RHS expression contains more than one matrix");
    res.resize(A_rows, A_cols);
    for (index_t i = 0; i < A_rows; ++i)
      for (index_t j = 0; j < A_cols; ++j)
        res(i, j) =
            (A.block(i * u_r, j * u_c, u_r, u_c).array() * B.array()).sum();
    return res;
  }

  index_t rows() const { return 1; }
  index_t cols() const { return 1; }

  void parse(gsExprHelper<Scalar> &evList) const {
    _u.parse(evList);
    _v.parse(evList);
  }

  const gsFeSpace<Scalar> &rowVar() const { return _u.rowVar(); }
  const gsFeSpace<Scalar> &colVar() const { return _u.colVar(); } // overwrite

  void print(std::ostream &os) const {
    os << "(";
    _u.print(os);
    os << " % ";
    _v.print(os);
    os << ")";
  }
};
template <typename E1, typename E2>
EIGEN_STRONG_INLINE frobenius_prod_expr<E1, E2> const
frobenius(_expr<E1> const &u, _expr<E2> const &v) {
  return frobenius_prod_expr<E1, E2>(u, v);
}

} // namespace gismo::expr
