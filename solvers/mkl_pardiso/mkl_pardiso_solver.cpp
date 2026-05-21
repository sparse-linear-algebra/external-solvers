#include "sparse_solver_interface_plugin.hpp"

#include <mkl.h>

#include <algorithm>
#include <array>
#include <complex>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using ssi::complex128_t;
using ssi::complex64_t;
using ssi::dtype_t;
using ssi::float32_t;
using ssi::float64_t;
using ssi::graph_orientation_t;
using ssi::int32_t;
using ssi::int64_t;
using ssi::itype_t;
using ssi::matrix_order_t;
using ssi::property_state_t;
using ssi::sparse_problem_properties_t;
using ssi::symmetric_storage_t;

class context_t;
class graph_t;
class sparse_matrix_t;
class symbolic_t;

std::size_t dtype_size(dtype_t dtype){
  switch(dtype){
    case dtype_t::fp32: return sizeof(float32_t);
    case dtype_t::fp64: return sizeof(float64_t);
    case dtype_t::c64: return sizeof(complex64_t);
    case dtype_t::c128: return sizeof(complex128_t);
  }
  throw std::invalid_argument("unknown dtype");
}

bool is_complex(dtype_t dtype){
  return dtype == dtype_t::c64 || dtype == dtype_t::c128;
}

bool known_true(property_state_t state){
  return state == property_state_t::known_true;
}

template<typename Index>
Index checked_index(int64_t value,const char* name){
  if(value < 0){
    throw std::out_of_range(std::string(name) + " is negative");
  }
  if constexpr(std::is_same_v<Index,int32_t>){
    if(value > std::numeric_limits<int32_t>::max()){
      throw std::out_of_range(std::string(name) + " does not fit int32_t");
    }
  }
  return static_cast<Index>(value);
}

int select_mtype(const sparse_problem_properties_t& props){
  const bool complex = is_complex(props.dtype);
  const bool symmetric_storage =
    props.symmetric_storage == symmetric_storage_t::full ||
    props.symmetric_storage == symmetric_storage_t::lower ||
    props.symmetric_storage == symmetric_storage_t::upper;
  const bool numeric_sym = known_true(props.numerically_symmetric);
  const bool structural_sym = known_true(props.structurally_symmetric);
  const bool positive_definite = known_true(props.positive_definite);

  if(!complex){
    if(symmetric_storage && numeric_sym){
      return positive_definite ? 2 : -2;
    }
    if(symmetric_storage && structural_sym){
      return 1;
    }
    return 11;
  }

  if(symmetric_storage && numeric_sym){
    return positive_definite ? 4 : -4;
  }
  return 13;
}

void validate_storage_for_mtype(
  symmetric_storage_t storage,
  int mtype){
  const bool symmetric_mode =
    mtype == 1 || mtype == 2 || mtype == -2 || mtype == 4 || mtype == -4;
  if((storage == symmetric_storage_t::lower ||
      storage == symmetric_storage_t::upper) && !symmetric_mode){
    throw std::invalid_argument(
      "triangular symmetric storage requires a symmetric or Hermitian PARDISO mode");
  }
}

void throw_pardiso_error(long long error,const char* operation){
  if(error == 0){
    return;
  }
  throw std::runtime_error(
    std::string("PARDISO ") + operation + " failed with error " +
    std::to_string(error));
}

class dense_matrix_t final : public ssi::matrix_t{
  public:
    dense_matrix_t(std::shared_ptr<ssi::context_t> context,dtype_t dtype) :
      ssi::matrix_t(std::move(context)),
      dtype_(dtype) {}

    int64_t nrows() const override{ return nrows_; }
    int64_t ncols() const override{ return ncols_; }
    dtype_t dtype() const override{ return dtype_; }

    void preallocate(int64_t nrows,int64_t ncols) override{
      if(nrows < 0 || ncols < 0){
        throw std::invalid_argument("negative dense matrix extent");
      }
      nrows_ = nrows;
      ncols_ = ncols;
      data_.assign(static_cast<std::size_t>(nrows*ncols)*dtype_size(dtype_),0);
    }

    void borrow_matrix_view(
      const ssi::placement_t&,
      const ssi::matrix_view_t& view) override{
      copy_from_view(view);
    }

    void build_from_host(
      std::function<void(ssi::matrix_view_t&)>& builder) override{
      if(nrows_ < 0 || ncols_ < 0){
        throw std::runtime_error("dense matrix must be preallocated before build");
      }
      auto view = mutable_view();
      builder(view);
    }

    void build_from_placement(
      std::function<void(const ssi::placement_t&,ssi::matrix_view_t&)>&) override{
      throw std::runtime_error("placement builds are not supported by mkl_pardiso");
    }

    void read_to_host(
      std::function<void(const ssi::matrix_view_t&)>& reader) const override{
      auto view = const_view();
      reader(view);
    }

    void read_to_placement(
      const ssi::placement_t&,
      std::function<void(const ssi::matrix_view_t&)>&) const override{
      throw std::runtime_error("placement reads are not supported by mkl_pardiso");
    }

  private:
    void copy_from_view(const ssi::matrix_view_t& view){
      if(view.dtype != dtype_){
        throw std::invalid_argument("dense matrix dtype mismatch");
      }
      preallocate(view.rend - view.rbeg,view.cend - view.cbeg);
      const auto rows = nrows_;
      const auto cols = ncols_;
      const auto elem = dtype_size(dtype_);
      const auto* src = const_data_pointer(view);
      for(int64_t col = 0; col < cols; ++col){
        for(int64_t row = 0; row < rows; ++row){
          const auto src_index = view.order == matrix_order_t::col_major
            ? row + view.ld*col
            : col + view.ld*row;
          const auto dst_index = row + rows*col;
          std::memcpy(
            data_.data() + static_cast<std::size_t>(dst_index)*elem,
            src + static_cast<std::size_t>(src_index)*elem,
            elem);
        }
      }
    }

    ssi::matrix_view_t mutable_view(){
      ssi::matrix_view_t view{
        matrix_order_t::col_major,
        dtype_,
        0,
        nrows_,
        0,
        ncols_,
        nrows_,
        {.fp32 = nullptr}
      };
      set_view_data(view,data_.data());
      return view;
    }

    ssi::matrix_view_t const_view() const{
      auto view = const_cast<dense_matrix_t*>(this)->mutable_view();
      return view;
    }

    static void set_view_data(ssi::matrix_view_t& view,void* data){
      if(view.dtype == dtype_t::fp32) view.d.fp32 = static_cast<float32_t*>(data);
      if(view.dtype == dtype_t::fp64) view.d.fp64 = static_cast<float64_t*>(data);
      if(view.dtype == dtype_t::c64) view.d.c64 = static_cast<complex64_t*>(data);
      if(view.dtype == dtype_t::c128) view.d.c128 = static_cast<complex128_t*>(data);
    }

    static const unsigned char* const_data_pointer(const ssi::matrix_view_t& view){
      if(view.dtype == dtype_t::fp32) return reinterpret_cast<const unsigned char*>(view.d.fp32);
      if(view.dtype == dtype_t::fp64) return reinterpret_cast<const unsigned char*>(view.d.fp64);
      if(view.dtype == dtype_t::c64) return reinterpret_cast<const unsigned char*>(view.d.c64);
      if(view.dtype == dtype_t::c128) return reinterpret_cast<const unsigned char*>(view.d.c128);
      throw std::invalid_argument("unknown dtype");
    }

    int64_t nrows_ = -1;
    int64_t ncols_ = -1;
    dtype_t dtype_;
    std::vector<unsigned char> data_;
};

class graph_t final :
  public ssi::graph_t,
  public std::enable_shared_from_this<graph_t>{
  public:
    graph_t(
      std::shared_ptr<ssi::context_t> context,
      itype_t itype,
      sparse_problem_properties_t properties) :
      ssi::graph_t(std::move(context)),
      itype_(itype),
      properties_(properties) {
      properties_.itype = itype;
    }

    itype_t itype() const override{ return itype_; }
    int64_t nrows() const override{ return nrows_; }
    int64_t ncols() const override{ return ncols_; }
    int64_t nedges() const override{ return nedges_; }

    void build_from_host(
      int64_t nrows,
      int64_t ncols,
      graph_orientation_t orientation,
      std::function<void(ssi::graph_count_builder_t&)>& count_builder,
      std::function<void(ssi::graph_edge_builder_t&)>& edge_builder) override{
      if(orientation != graph_orientation_t::row){
        throw std::invalid_argument("mkl_pardiso requires row-oriented graphs");
      }
      if(nrows < 0 || ncols < 0){
        throw std::invalid_argument("negative graph extent");
      }
      std::vector<int64_t> counts(static_cast<std::size_t>(nrows));
      ssi::graph_count_builder_t count_view{
        orientation,
        itype_,
        nrows,
        ncols,
        0,
        nrows,
        {.i64 = counts.data()}
      };
      if(itype_ == itype_t::i32){
        std::vector<int32_t> counts32(static_cast<std::size_t>(nrows));
        count_view.counts.i32 = counts32.data();
        count_builder(count_view);
        for(int64_t i = 0; i < nrows; ++i) counts[static_cast<std::size_t>(i)] = counts32[static_cast<std::size_t>(i)];
      }else{
        count_builder(count_view);
      }

      nrows_ = nrows;
      ncols_ = ncols;
      orientation_ = orientation;
      nedges_ = std::accumulate(counts.begin(),counts.end(),int64_t{0});
      allocate_offsets_from_counts(counts);
      allocate_ids();

      ssi::graph_edge_builder_t edge_view{
        orientation_,
        itype_,
        nrows_,
        ncols_,
        0,
        nrows_,
        {.i32 = nullptr},
        {.i32 = nullptr}
      };
      if(itype_ == itype_t::i32){
        edge_view.offsets.i32 = offsets32_.data();
        edge_view.ids.i32 = ids32_.data();
      }else{
        edge_view.offsets.i64 = offsets64_.data();
        edge_view.ids.i64 = ids64_.data();
      }
      edge_builder(edge_view);
      canonicalize_and_validate_rows();
    }

    void borrow_compressed_graph_view(
      const ssi::compressed_graph_view_t& view) override{
      if(view.orientation != graph_orientation_t::row){
        throw std::invalid_argument("mkl_pardiso requires row-oriented graphs");
      }
      if(view.itype != itype_){
        throw std::invalid_argument("borrowed graph index type mismatch");
      }
      if(view.beg != 0 || view.end != view.nrows){
        throw std::invalid_argument("mkl_pardiso requires complete CSR graph views");
      }
      nrows_ = view.nrows;
      ncols_ = view.ncols;
      orientation_ = view.orientation;
      const int64_t major = nrows_;
      nedges_ = itype_ == itype_t::i32
        ? view.offsets.i32[major]
        : view.offsets.i64[major];
      if(itype_ == itype_t::i32){
        offsets32_.resize(static_cast<std::size_t>(major + 1));
        ids32_.resize(static_cast<std::size_t>(nedges_));
        std::copy_n(view.offsets.i32,major + 1,offsets32_.data());
        std::copy_n(view.ids.i32,nedges_,ids32_.data());
      }else{
        offsets64_.resize(static_cast<std::size_t>(major + 1));
        ids64_.resize(static_cast<std::size_t>(nedges_));
        std::copy_n(view.offsets.i64,major + 1,offsets64_.data());
        std::copy_n(view.ids.i64,nedges_,ids64_.data());
      }
      canonicalize_and_validate_rows();
    }

    std::shared_ptr<ssi::sparse_matrix_t> make_sparse_matrix() override;
    std::shared_ptr<ssi::symbolic_t> make_symbolic_analysis() override;

    const sparse_problem_properties_t& properties() const{ return properties_; }
    const void* offsets_data() const{
      return itype_ == itype_t::i32
        ? static_cast<const void*>(offsets32_.data())
        : static_cast<const void*>(offsets64_.data());
    }
    void* offsets_data(){
      return itype_ == itype_t::i32
        ? static_cast<void*>(offsets32_.data())
        : static_cast<void*>(offsets64_.data());
    }
    const void* ids_data() const{
      return itype_ == itype_t::i32
        ? static_cast<const void*>(ids32_.data())
        : static_cast<const void*>(ids64_.data());
    }
    void* ids_data(){
      return itype_ == itype_t::i32
        ? static_cast<void*>(ids32_.data())
        : static_cast<void*>(ids64_.data());
    }

  private:
    void allocate_offsets_from_counts(const std::vector<int64_t>& counts){
      if(itype_ == itype_t::i32){
        offsets32_.assign(counts.size() + 1,0);
        int64_t running = 0;
        for(std::size_t i = 0; i < counts.size(); ++i){
          if(counts[i] < 0) throw std::invalid_argument("negative graph edge count");
          running += counts[i];
          offsets32_[i + 1] = checked_index<int32_t>(running,"graph offset");
        }
      }else{
        offsets64_.assign(counts.size() + 1,0);
        int64_t running = 0;
        for(std::size_t i = 0; i < counts.size(); ++i){
          if(counts[i] < 0) throw std::invalid_argument("negative graph edge count");
          running += counts[i];
          offsets64_[i + 1] = running;
        }
      }
    }

    void allocate_ids(){
      if(itype_ == itype_t::i32){
        ids32_.assign(static_cast<std::size_t>(nedges_),0);
      }else{
        ids64_.assign(static_cast<std::size_t>(nedges_),0);
      }
    }

    template<typename Index>
    void canonicalize_rows(std::vector<Index>& offsets,std::vector<Index>& ids){
      if(offsets.empty()) return;
      if(offsets.front() != 0){
        throw std::invalid_argument("CSR offsets must start at zero");
      }
      for(int64_t row = 0; row < nrows_; ++row){
        const auto beg = offsets[static_cast<std::size_t>(row)];
        const auto end = offsets[static_cast<std::size_t>(row + 1)];
        if(beg > end || beg < 0 || end > nedges_){
          throw std::invalid_argument("invalid CSR offsets");
        }
        std::vector<Index> row_ids(
          ids.begin() + beg,
          ids.begin() + end);
        std::sort(row_ids.begin(),row_ids.end());
        for(auto it = row_ids.begin(); it != row_ids.end(); ++it){
          if(*it < 0 || *it >= ncols_){
            throw std::out_of_range("CSR column id is outside matrix extent");
          }
          if(it != row_ids.begin() && *it == *(it - 1)){
            throw std::invalid_argument("duplicate CSR column id");
          }
        }
      }
    }

    void canonicalize_and_validate_rows(){
      if(nrows_ != ncols_){
        throw std::invalid_argument("mkl_pardiso requires square matrices");
      }
      if(itype_ == itype_t::i32){
        canonicalize_rows(offsets32_,ids32_);
      }else{
        canonicalize_rows(offsets64_,ids64_);
      }
    }

    itype_t itype_;
    sparse_problem_properties_t properties_;
    int64_t nrows_ = 0;
    int64_t ncols_ = 0;
    int64_t nedges_ = 0;
    graph_orientation_t orientation_ = graph_orientation_t::row;
    std::vector<int32_t> offsets32_;
    std::vector<int64_t> offsets64_;
    std::vector<int32_t> ids32_;
    std::vector<int64_t> ids64_;
};

class sparse_matrix_t final : public ssi::sparse_matrix_t{
  public:
    explicit sparse_matrix_t(std::shared_ptr<graph_t> graph) :
      ssi::sparse_matrix_t(graph),
      graph_(std::move(graph)) {}

    int64_t nrows() const override{ return graph_->nrows(); }
    int64_t ncols() const override{ return graph_->ncols(); }
    dtype_t dtype() const override{ return dtype_; }

    void build_from_host(
      dtype_t dtype,
      graph_orientation_t orientation,
      std::function<void(ssi::sparse_value_builder_t&)>& builder) override{
      if(orientation != graph_orientation_t::row){
        throw std::invalid_argument("mkl_pardiso requires row-oriented sparse values");
      }
      dtype_ = dtype;
      values_.assign(
        static_cast<std::size_t>(graph_->nedges())*dtype_size(dtype_),
        0);
      auto view = value_builder();
      builder(view);
    }

    void read_to_host(
      graph_orientation_t orientation,
      std::function<void(const ssi::sparse_value_builder_t&)>& reader) const override{
      if(orientation != graph_orientation_t::row){
        throw std::invalid_argument("mkl_pardiso stores row-oriented sparse values");
      }
      auto view = const_cast<sparse_matrix_t*>(this)->value_builder();
      reader(view);
    }

    void borrow_sparse_values_view(
      const ssi::sparse_values_view_t& view) override{
      if(view.nedges != graph_->nedges()){
        throw std::invalid_argument("sparse value count does not match graph");
      }
      dtype_ = view.dtype;
      values_.resize(static_cast<std::size_t>(view.nedges)*dtype_size(dtype_));
      const void* src = nullptr;
      if(dtype_ == dtype_t::fp32) src = view.values.fp32;
      if(dtype_ == dtype_t::fp64) src = view.values.fp64;
      if(dtype_ == dtype_t::c64) src = view.values.c64;
      if(dtype_ == dtype_t::c128) src = view.values.c128;
      std::memcpy(values_.data(),src,values_.size());
    }

    const void* values_data() const{ return values_.data(); }
    const unsigned char* value_bytes() const{ return values_.data(); }

  private:
    ssi::sparse_value_builder_t value_builder(){
      ssi::sparse_value_builder_t out{
        graph_orientation_t::row,
        graph_->itype(),
        dtype_,
        graph_->nrows(),
        graph_->ncols(),
        0,
        graph_->nrows(),
        {.i32 = nullptr},
        {.i32 = nullptr},
        {.fp32 = nullptr}
      };
      if(graph_->itype() == itype_t::i32){
        out.offsets.i32 = static_cast<const int32_t*>(graph_->offsets_data());
        out.ids.i32 = static_cast<const int32_t*>(graph_->ids_data());
      }else{
        out.offsets.i64 = static_cast<const int64_t*>(graph_->offsets_data());
        out.ids.i64 = static_cast<const int64_t*>(graph_->ids_data());
      }
      if(dtype_ == dtype_t::fp32) out.values.fp32 = reinterpret_cast<float32_t*>(values_.data());
      if(dtype_ == dtype_t::fp64) out.values.fp64 = reinterpret_cast<float64_t*>(values_.data());
      if(dtype_ == dtype_t::c64) out.values.c64 = reinterpret_cast<complex64_t*>(values_.data());
      if(dtype_ == dtype_t::c128) out.values.c128 = reinterpret_cast<complex128_t*>(values_.data());
      return out;
    }

    std::shared_ptr<graph_t> graph_;
    dtype_t dtype_ = dtype_t::fp64;
    std::vector<unsigned char> values_;
};

class symbolic_t final :
  public ssi::symbolic_t,
  public std::enable_shared_from_this<symbolic_t>{
  public:
    explicit symbolic_t(std::shared_ptr<graph_t> graph) :
      ssi::symbolic_t(graph),
      graph_(std::move(graph)) {}

    std::shared_ptr<ssi::numeric_factorization_t>
    make_numeric_factorization(std::shared_ptr<ssi::sparse_matrix_t> matrix) override;

    std::shared_ptr<graph_t> graph_ptr() const{ return graph_; }

  private:
    std::shared_ptr<graph_t> graph_;
};

template<typename Index>
void pardiso_call(
  std::array<void*,64>& pt,
  Index& maxfct,
  Index& mnum,
  Index& mtype,
  Index& phase,
  Index& n,
  const void* a,
  const Index* ia,
  const Index* ja,
  Index* perm,
  Index& nrhs,
  std::array<Index,64>& iparm,
  Index& msglvl,
  void* b,
  void* x,
  Index& error){
  if constexpr(std::is_same_v<Index,int32_t>){
    pardiso(
      pt.data(),&maxfct,&mnum,&mtype,&phase,&n,a,ia,ja,perm,&nrhs,
      iparm.data(),&msglvl,b,x,&error);
  }else{
    pardiso_64(
      pt.data(),&maxfct,&mnum,&mtype,&phase,&n,a,ia,ja,perm,&nrhs,
      iparm.data(),&msglvl,b,x,&error);
  }
}

class numeric_factorization_t final : public ssi::numeric_factorization_t{
  public:
    numeric_factorization_t(
      std::shared_ptr<symbolic_t> symbolic,
      std::shared_ptr<sparse_matrix_t> matrix) :
      ssi::numeric_factorization_t(symbolic,matrix),
      symbolic_(std::move(symbolic)),
      matrix_(std::move(matrix)) {
      factor();
    }

    ~numeric_factorization_t() override{
      try{
        release();
      }catch(...){
      }
    }

    dtype_t dtype() const override{ return matrix_->dtype(); }

    void solve(const ssi::matrix_t& rhs,ssi::matrix_t& solution) const override{
      if(rhs.dtype() != dtype() || solution.dtype() != dtype()){
        throw std::invalid_argument("dense solve dtype mismatch");
      }
      if(rhs.nrows() != symbolic_->graph_ptr()->nrows()){
        throw std::invalid_argument("RHS row count does not match factorization");
      }
      solution.preallocate(rhs.nrows(),rhs.ncols());

      std::vector<unsigned char> b;
      read_dense(rhs,b);
      std::vector<unsigned char> x(b.size(),0);
      solve_impl(b.data(),x.data(),rhs.ncols());
      write_dense(solution,x);
    }

  private:
    template<typename Index>
    void init_iparm(std::array<Index,64>& iparm,Index){
      iparm.fill(0);
      iparm[0] = 1;    // Use explicit parameter values.
      iparm[1] = 2;    // METIS fill-reducing ordering.
      iparm[7] = 2;    // Iterative refinement steps.
      iparm[9] = 13;   // Pivot perturbation.
      iparm[10] = 1;   // Scaling.
      iparm[12] = 1;   // Matching.
      iparm[17] = -1;  // Report factor nonzeros.
      iparm[18] = -1;  // Report factorization Mflops.
      iparm[34] = 0;   // PARDISO receives one-based CSR translated from SSI.
    }

    void factor(){
      const auto& props = symbolic_->graph_ptr()->properties();
      mtype_ = select_mtype(props);
      validate_storage_for_mtype(props.symmetric_storage,mtype_);
      pardiso_uses_i64_ = requires_pardiso_i64();
      if(!pardiso_uses_i64_){
        factor_impl<int32_t>();
      }else{
        factor_impl<long long>();
      }
    }

    template<typename Index>
    void factor_impl(){
      if constexpr(std::is_same_v<Index,int32_t>){
        refresh_pardiso_i32_csr();
      }
      if constexpr(std::is_same_v<Index,long long>){
        refresh_pardiso_i64_csr();
      }
      validate_pardiso_csr<Index>();
      auto& iparm = iparm_for<Index>();
      Index maxfct = 1;
      Index mnum = 1;
      Index mtype = static_cast<Index>(mtype_);
      Index phase = 11;
      Index n = checked_index<Index>(symbolic_->graph_ptr()->nrows(),"matrix dimension");
      Index nrhs = 1;
      Index msglvl = 0;
      Index error = 0;
      std::vector<Index> perm(static_cast<std::size_t>(n),0);
      init_iparm(iparm,mtype);

      pardiso_call<Index>(
        pt_,maxfct,mnum,mtype,phase,n,pardiso_values_.data(),
        pardiso_offsets<Index>(),pardiso_ids<Index>(),perm.data(),nrhs,iparm,msglvl,
        nullptr,nullptr,error);
      throw_pardiso_error(error,"analysis");

      phase = 22;
      pardiso_call<Index>(
        pt_,maxfct,mnum,mtype,phase,n,pardiso_values_.data(),
        pardiso_offsets<Index>(),pardiso_ids<Index>(),perm.data(),nrhs,iparm,msglvl,
        nullptr,nullptr,error);
      throw_pardiso_error(error,"numeric factorization");
      factored_ = true;
    }

    void solve_impl(void* b,void* x,int64_t nrhs_value) const{
      if(pardiso_uses_i64_){
        solve_impl_t<long long>(b,x,nrhs_value);
      }else{
        solve_impl_t<int32_t>(b,x,nrhs_value);
      }
    }

    template<typename Index>
    void solve_impl_t(void* b,void* x,int64_t nrhs_value) const{
      auto& iparm = const_cast<numeric_factorization_t*>(this)->iparm_for<Index>();
      Index maxfct = 1;
      Index mnum = 1;
      Index mtype = static_cast<Index>(mtype_);
      Index phase = 33;
      Index n = checked_index<Index>(symbolic_->graph_ptr()->nrows(),"matrix dimension");
      Index nrhs = checked_index<Index>(nrhs_value,"number of right hand sides");
      Index msglvl = 0;
      Index error = 0;
      std::vector<Index> perm(static_cast<std::size_t>(n),0);
      pardiso_call<Index>(
        const_cast<std::array<void*,64>&>(pt_),maxfct,mnum,mtype,phase,n,
        pardiso_values_.data(),pardiso_offsets<Index>(),pardiso_ids<Index>(),perm.data(),nrhs,
        iparm,msglvl,b,x,error);
      throw_pardiso_error(error,"solve");
    }

    void release(){
      if(!factored_){
        return;
      }
      if(pardiso_uses_i64_){
        release_impl<long long>();
      }else{
        release_impl<int32_t>();
      }
      factored_ = false;
    }

    template<typename Index>
    void release_impl(){
      auto& iparm = iparm_for<Index>();
      Index maxfct = 1;
      Index mnum = 1;
      Index mtype = static_cast<Index>(mtype_);
      Index phase = -1;
      Index n = checked_index<Index>(symbolic_->graph_ptr()->nrows(),"matrix dimension");
      Index nrhs = 1;
      Index msglvl = 0;
      Index error = 0;
      std::vector<Index> perm(static_cast<std::size_t>(n),0);
      pardiso_call<Index>(
        pt_,maxfct,mnum,mtype,phase,n,nullptr,pardiso_offsets<Index>(),pardiso_ids<Index>(),
        perm.data(),nrhs,iparm,msglvl,nullptr,nullptr,error);
    }

    template<typename Index>
    std::array<Index,64>& iparm_for(){
      if constexpr(std::is_same_v<Index,int32_t>){
        return iparm32_;
      }else{
        return iparm64_;
      }
    }

    template<typename Index>
    const Index* pardiso_offsets() const{
      if constexpr(std::is_same_v<Index,int32_t>){
        return offsets32_pardiso_.data();
      }else{
        return offsets64ll_.data();
      }
    }

    template<typename Index>
    const Index* pardiso_ids() const{
      if constexpr(std::is_same_v<Index,int32_t>){
        return ids32_pardiso_.data();
      }else{
        return ids64ll_.data();
      }
    }

    template<typename Index>
    void validate_pardiso_csr() const{
      const auto n = checked_index<Index>(
        symbolic_->graph_ptr()->nrows(),
        "matrix dimension");
      const auto nnz = checked_index<Index>(
        pardiso_nnz_,
        "nonzero count");
      const auto* ia = pardiso_offsets<Index>();
      const auto* ja = pardiso_ids<Index>();
      if(ia == nullptr || ja == nullptr){
        throw std::runtime_error("null CSR data");
      }
      if(ia[0] != 1){
        throw std::invalid_argument("PARDISO CSR row offsets must start at one");
      }
      if(ia[n] != nnz + 1){
        throw std::invalid_argument("CSR final row offset does not match nonzero count");
      }
      for(Index row = 0; row < n; ++row){
        if(ia[row] > ia[row + 1]){
          throw std::invalid_argument("CSR row offsets must be nondecreasing");
        }
        if(ia[row] < 1 || ia[row + 1] > nnz + 1){
          throw std::out_of_range("CSR row offset is outside nonzero range");
        }
        for(Index pos = ia[row]; pos < ia[row + 1]; ++pos){
          if(ja[pos - 1] < 1 || ja[pos - 1] > n){
            throw std::out_of_range("CSR column id is outside matrix extent");
          }
        }
      }
    }

    bool requires_pardiso_i64() const{
      const auto graph = symbolic_->graph_ptr();
      if(graph->nrows() > std::numeric_limits<int32_t>::max() ||
         graph->nedges() > std::numeric_limits<int32_t>::max()){
        return true;
      }
      if(graph->itype() == itype_t::i32){
        return false;
      }
      const auto* offsets = static_cast<const int64_t*>(graph->offsets_data());
      const auto* ids = static_cast<const int64_t*>(graph->ids_data());
      const auto n = graph->nrows();
      const auto nnz = graph->nedges();
      for(int64_t i = 0; i <= n; ++i){
        if(offsets[i] > std::numeric_limits<int32_t>::max()){
          return true;
        }
      }
      for(int64_t i = 0; i < nnz; ++i){
        if(ids[i] > std::numeric_limits<int32_t>::max()){
          return true;
        }
      }
      return false;
    }

    void refresh_pardiso_i32_csr(){
      const auto graph = symbolic_->graph_ptr();
      const auto n = graph->nrows();
      const auto graph_nnz = graph->nedges();
      offsets32_pardiso_.resize(static_cast<std::size_t>(n + 1));
      ids32_pardiso_.clear();
      pardiso_values_.clear();
      const auto elem = dtype_size(matrix_->dtype());
      const auto* values = matrix_->value_bytes();
      const auto symmetric_mode = uses_symmetric_pardiso_storage();
      auto include_entry = [&](int64_t row,int64_t col){
        if(!symmetric_mode){
          return true;
        }
        const auto storage = symbolic_->graph_ptr()->properties().symmetric_storage;
        if(storage == symmetric_storage_t::lower){
          return col <= row;
        }
        return col >= row;
      };
      int64_t out_pos = 0;
      offsets32_pardiso_[0] = 1;
      if(graph->itype() == itype_t::i32){
        const auto* offsets = static_cast<const int32_t*>(graph->offsets_data());
        const auto* ids = static_cast<const int32_t*>(graph->ids_data());
        for(int64_t row = 0; row < n; ++row){
          std::vector<std::pair<int64_t,int64_t>> row_entries;
          for(int64_t pos = offsets[row]; pos < offsets[row + 1]; ++pos){
            const int64_t col = ids[pos];
            if(!include_entry(row,col)){
              continue;
            }
            row_entries.emplace_back(col,pos);
          }
          std::sort(row_entries.begin(),row_entries.end());
          for(const auto& [col,pos] : row_entries){
            ids32_pardiso_.push_back(checked_index<int32_t>(col + 1,"CSR column id"));
            append_pardiso_value(values + static_cast<std::size_t>(pos)*elem,elem);
            ++out_pos;
          }
          offsets32_pardiso_[static_cast<std::size_t>(row + 1)] =
            checked_index<int32_t>(out_pos + 1,"CSR row offset");
        }
        pardiso_nnz_ = out_pos;
        return;
      }
      const auto* offsets = static_cast<const int64_t*>(graph->offsets_data());
      const auto* ids = static_cast<const int64_t*>(graph->ids_data());
      for(int64_t row = 0; row < n; ++row){
        std::vector<std::pair<int64_t,int64_t>> row_entries;
        for(int64_t pos = offsets[row]; pos < offsets[row + 1]; ++pos){
          const int64_t col = ids[pos];
          if(!include_entry(row,col)){
            continue;
          }
          row_entries.emplace_back(col,pos);
        }
        std::sort(row_entries.begin(),row_entries.end());
        for(const auto& [col,pos] : row_entries){
          ids32_pardiso_.push_back(checked_index<int32_t>(col + 1,"CSR column id"));
          append_pardiso_value(values + static_cast<std::size_t>(pos)*elem,elem);
          ++out_pos;
        }
        offsets32_pardiso_[static_cast<std::size_t>(row + 1)] =
          checked_index<int32_t>(out_pos + 1,"CSR row offset");
      }
      pardiso_nnz_ = out_pos;
      if(graph_nnz > 0 && pardiso_nnz_ == 0){
        throw std::invalid_argument("PARDISO CSR has no entries after storage conversion");
      }
    }

    void refresh_pardiso_i64_csr(){
      const auto n = symbolic_->graph_ptr()->nrows();
      const auto graph_nnz = symbolic_->graph_ptr()->nedges();
      const auto* offsets = static_cast<const int64_t*>(symbolic_->graph_ptr()->offsets_data());
      const auto* ids = static_cast<const int64_t*>(symbolic_->graph_ptr()->ids_data());
      offsets64ll_.resize(static_cast<std::size_t>(n + 1));
      ids64ll_.clear();
      pardiso_values_.clear();
      const auto elem = dtype_size(matrix_->dtype());
      const auto* values = matrix_->value_bytes();
      const auto symmetric_mode = uses_symmetric_pardiso_storage();
      auto include_entry = [&](int64_t row,int64_t col){
        if(!symmetric_mode){
          return true;
        }
        const auto storage = symbolic_->graph_ptr()->properties().symmetric_storage;
        if(storage == symmetric_storage_t::lower){
          return col <= row;
        }
        return col >= row;
      };
      int64_t out_pos = 0;
      offsets64ll_[0] = 1;
      for(int64_t row = 0; row < n; ++row){
        std::vector<std::pair<int64_t,int64_t>> row_entries;
        for(int64_t pos = offsets[row]; pos < offsets[row + 1]; ++pos){
          const int64_t col = ids[pos];
          if(!include_entry(row,col)){
            continue;
          }
          row_entries.emplace_back(col,pos);
        }
        std::sort(row_entries.begin(),row_entries.end());
        for(const auto& [col,pos] : row_entries){
          ids64ll_.push_back(col + 1);
          append_pardiso_value(values + static_cast<std::size_t>(pos)*elem,elem);
          ++out_pos;
        }
        offsets64ll_[static_cast<std::size_t>(row + 1)] = out_pos + 1;
      }
      pardiso_nnz_ = out_pos;
      if(graph_nnz > 0 && pardiso_nnz_ == 0){
        throw std::invalid_argument("PARDISO CSR has no entries after storage conversion");
      }
    }

    bool uses_symmetric_pardiso_storage() const{
      return mtype_ == 1 || mtype_ == 2 || mtype_ == -2 ||
        mtype_ == 4 || mtype_ == -4;
    }

    void append_pardiso_value(const unsigned char* src,std::size_t elem){
      const auto old = pardiso_values_.size();
      pardiso_values_.resize(old + elem);
      std::memcpy(pardiso_values_.data() + old,src,elem);
    }

    void read_dense(const ssi::matrix_t& matrix,std::vector<unsigned char>& out) const{
      out.assign(
        static_cast<std::size_t>(matrix.nrows()*matrix.ncols())*dtype_size(matrix.dtype()),
        0);
      std::function<void(const ssi::matrix_view_t&)> reader =
        [&](const ssi::matrix_view_t& view){
          copy_dense_view_to_col_major(view,out.data());
        };
      matrix.read_to_host(reader);
    }

    void write_dense(ssi::matrix_t& matrix,const std::vector<unsigned char>& data) const{
      std::function<void(ssi::matrix_view_t&)> writer =
        [&](ssi::matrix_view_t& view){
          copy_col_major_to_dense_view(data.data(),view);
        };
      matrix.build_from_host(writer);
    }

    static void copy_dense_view_to_col_major(
      const ssi::matrix_view_t& view,
      void* out){
      const auto rows = view.rend - view.rbeg;
      const auto cols = view.cend - view.cbeg;
      const auto elem = dtype_size(view.dtype);
      for(int64_t col = 0; col < cols; ++col){
        for(int64_t row = 0; row < rows; ++row){
          const auto src_index = view.order == matrix_order_t::col_major
            ? row + view.ld*col
            : col + view.ld*row;
          const auto dst_index = row + rows*col;
          std::memcpy(
            static_cast<unsigned char*>(out) + static_cast<std::size_t>(dst_index)*elem,
            data_pointer(view) + static_cast<std::size_t>(src_index)*elem,
            elem);
        }
      }
    }

    static void copy_col_major_to_dense_view(
      const void* in,
      ssi::matrix_view_t& view){
      const auto rows = view.rend - view.rbeg;
      const auto cols = view.cend - view.cbeg;
      const auto elem = dtype_size(view.dtype);
      for(int64_t col = 0; col < cols; ++col){
        for(int64_t row = 0; row < rows; ++row){
          const auto src_index = row + rows*col;
          const auto dst_index = view.order == matrix_order_t::col_major
            ? row + view.ld*col
            : col + view.ld*row;
          std::memcpy(
            data_pointer(view) + static_cast<std::size_t>(dst_index)*elem,
            static_cast<const unsigned char*>(in) + static_cast<std::size_t>(src_index)*elem,
            elem);
        }
      }
    }

    static unsigned char* data_pointer(ssi::matrix_view_t& view){
      if(view.dtype == dtype_t::fp32) return reinterpret_cast<unsigned char*>(view.d.fp32);
      if(view.dtype == dtype_t::fp64) return reinterpret_cast<unsigned char*>(view.d.fp64);
      if(view.dtype == dtype_t::c64) return reinterpret_cast<unsigned char*>(view.d.c64);
      if(view.dtype == dtype_t::c128) return reinterpret_cast<unsigned char*>(view.d.c128);
      throw std::invalid_argument("unknown dtype");
    }

    static const unsigned char* data_pointer(const ssi::matrix_view_t& view){
      if(view.dtype == dtype_t::fp32) return reinterpret_cast<const unsigned char*>(view.d.fp32);
      if(view.dtype == dtype_t::fp64) return reinterpret_cast<const unsigned char*>(view.d.fp64);
      if(view.dtype == dtype_t::c64) return reinterpret_cast<const unsigned char*>(view.d.c64);
      if(view.dtype == dtype_t::c128) return reinterpret_cast<const unsigned char*>(view.d.c128);
      throw std::invalid_argument("unknown dtype");
    }

    std::shared_ptr<symbolic_t> symbolic_;
    std::shared_ptr<sparse_matrix_t> matrix_;
    mutable std::array<void*,64> pt_{};
    std::array<int32_t,64> iparm32_{};
    std::array<long long,64> iparm64_{};
    mutable std::vector<int32_t> offsets32_pardiso_;
    mutable std::vector<int32_t> ids32_pardiso_;
    mutable std::vector<long long> offsets64ll_;
    mutable std::vector<long long> ids64ll_;
    mutable std::vector<unsigned char> pardiso_values_;
    int64_t pardiso_nnz_ = 0;
    int mtype_ = 11;
    bool pardiso_uses_i64_ = false;
    bool factored_ = false;
};

std::shared_ptr<ssi::sparse_matrix_t> graph_t::make_sparse_matrix(){
  return std::make_shared<sparse_matrix_t>(shared_from_this());
}

std::shared_ptr<ssi::symbolic_t> graph_t::make_symbolic_analysis(){
  return std::make_shared<symbolic_t>(shared_from_this());
}

std::shared_ptr<ssi::numeric_factorization_t>
symbolic_t::make_numeric_factorization(std::shared_ptr<ssi::sparse_matrix_t> matrix){
  auto typed_matrix = std::dynamic_pointer_cast<sparse_matrix_t>(std::move(matrix));
  if(typed_matrix == nullptr){
    throw std::invalid_argument("mkl_pardiso symbolic analysis requires an mkl_pardiso sparse matrix");
  }
  return std::make_shared<numeric_factorization_t>(
    shared_from_this(),
    std::move(typed_matrix));
}

class sparse_problem_t final : public ssi::sparse_problem_t{
  public:
    sparse_problem_t(
      std::shared_ptr<ssi::context_t> context,
      sparse_problem_properties_t properties) :
      ssi::sparse_problem_t(std::move(context),properties) {}

    std::shared_ptr<ssi::graph_t> make_graph() override;
    std::shared_ptr<ssi::sparse_matrix_t> make_sparse_matrix() override{
      if(matrix_ == nullptr){
        matrix_ = make_graph()->make_sparse_matrix();
      }
      return matrix_;
    }
    std::shared_ptr<ssi::symbolic_t> make_symbolic_analysis() override{
      if(symbolic_ == nullptr){
        symbolic_ = make_graph()->make_symbolic_analysis();
      }
      return symbolic_;
    }

  private:
    std::shared_ptr<ssi::graph_t> graph_;
    std::shared_ptr<ssi::sparse_matrix_t> matrix_;
    std::shared_ptr<ssi::symbolic_t> symbolic_;
};

class context_t final :
  public ssi::context_t,
  public std::enable_shared_from_this<context_t>{
  public:
    std::shared_ptr<ssi::matrix_t> make_matrix(dtype_t dtype) override{
      return std::make_shared<dense_matrix_t>(shared_from_this(),dtype);
    }

    std::shared_ptr<ssi::graph_t> make_graph(itype_t itype) override{
      sparse_problem_properties_t props;
      props.itype = itype;
      return std::make_shared<graph_t>(shared_from_this(),itype,props);
    }

    std::shared_ptr<ssi::sparse_problem_t> make_sparse_problem(
      const sparse_problem_properties_t& properties) override{
      return std::make_shared<sparse_problem_t>(shared_from_this(),properties);
    }
};

std::shared_ptr<ssi::graph_t> sparse_problem_t::make_graph(){
  if(graph_ == nullptr){
    const auto props = properties();
    graph_ = std::make_shared<graph_t>(context_ptr(),props.itype,props);
  }
  return graph_;
}

std::shared_ptr<ssi::context_t> create_context(){
  return std::make_shared<context_t>();
}

}  // namespace

SSI_EXPORT_PLUGIN(create_context)
