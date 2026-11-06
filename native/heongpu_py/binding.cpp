// Python binding for the CKKS operations QSH's backend protocol needs (qsh.fhe.backends.CKKSBackend).
#include <heongpu/heongpu.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <complex>
#include <memory>
#include <cuda_runtime.h>
#include <vector>

namespace py = pybind11;
constexpr auto Scheme = heongpu::Scheme::CKKS;
using Ct = heongpu::Ciphertext<Scheme>;
// Vectors cross as NumPy arrays (a list is converted on the way in). As Python lists of 4,096 complex numbers they took
// longer to build than the GPU took to mask or decode them (Round 86).
using Reals = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Slots = py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>;

static std::vector<double> reals(const Reals& a) { return std::vector<double>(a.data(), a.data() + a.size()); }

class Backend
{
  public:
    Backend(size_t ring, const std::vector<int>& q_bits, const std::vector<int>& p_bits, int scale_bits)
        : context_(heongpu::GenHEContext<Scheme>(heongpu::sec_level_type::sec128)),
          scale_(std::pow(2.0, scale_bits)), ring_(ring)
    {
        context_->set_poly_modulus_degree(ring);
        context_->set_coeff_modulus_bit_sizes(q_bits, p_bits);
        context_->generate();
        keygen_ = std::make_unique<heongpu::HEKeyGenerator<Scheme>>(context_);
        secret_ = std::make_unique<heongpu::Secretkey<Scheme>>(context_);
        keygen_->generate_secret_key(*secret_);
        public_ = std::make_unique<heongpu::Publickey<Scheme>>(context_);
        keygen_->generate_public_key(*public_, *secret_);
        galois_ = std::make_unique<heongpu::Galoiskey<Scheme>>(context_);
        keygen_->generate_galois_key(*galois_, *secret_);
        encoder_ = std::make_unique<heongpu::HEEncoder<Scheme>>(context_);
        encryptor_ = std::make_unique<heongpu::HEEncryptor<Scheme>>(context_, *public_);
        decryptor_ = std::make_unique<heongpu::HEDecryptor<Scheme>>(context_, *secret_);
        ops_ = std::make_unique<heongpu::HEArithmeticOperator<Scheme>>(context_, *encoder_);
    }

    size_t slot_count() const { return ring_ / 2; }

    std::shared_ptr<Ct> encrypt(const Reals& v)
    {
        heongpu::Plaintext<Scheme> p(context_);
        encoder_->encode(p, reals(v), scale_);
        auto c = std::make_shared<Ct>(context_);
        encryptor_->encrypt(*c, p);
        return c;
    }

    Slots decrypt_slots(const std::shared_ptr<Ct>& c)
    {
        heongpu::Plaintext<Scheme> p(context_);
        decryptor_->decrypt(p, *c);
        std::vector<Complex64> out;
        encoder_->decode(out, p);
        Slots r(static_cast<py::ssize_t>(out.size()));
        auto* d = r.mutable_data();
        for (size_t i = 0; i < out.size(); i++) d[i] = {out[i].real(), out[i].imag()};
        return r;
    }

    std::shared_ptr<Ct> add(const std::shared_ptr<Ct>& a, const std::shared_ptr<Ct>& b)
    {
        auto c = std::make_shared<Ct>(*a);
        ops_->add_inplace(*c, *b);
        return c;
    }

    // Encrypted under the public key at the ciphertext's own depth and scale, then added: a fresh
    // encryption re-randomises the ciphertext, which a plaintext addition would not.
    std::shared_ptr<Ct> add_plain_slots(const std::shared_ptr<Ct>& a, const Slots& values)
    {
        const auto* v = values.data();
        std::vector<Complex64> m(static_cast<size_t>(values.size()));
        for (size_t i = 0; i < m.size(); i++) m[i] = Complex64(v[i].real(), v[i].imag());
        heongpu::Plaintext<Scheme> p(context_);
        encoder_->encode(p, m, a->scale());
        Ct mask(context_);
        encryptor_->encrypt(mask, p);
        for (int d = 0; d < a->depth(); d++) ops_->mod_drop_inplace(mask);
        auto c = std::make_shared<Ct>(*a);
        ops_->add_inplace(*c, mask);
        return c;
    }

    // w . x in slot 0: a plaintext multiply, a rescale, and a rotate-and-add over the next power of two.
    std::shared_ptr<Ct> dot_plain(const std::shared_ptr<Ct>& a, const Reals& weights)
    {
        const std::vector<double> w = reals(weights);
        heongpu::Plaintext<Scheme> p(context_);
        encoder_->encode(p, w, scale_);
        auto c = std::make_shared<Ct>(*a);
        for (int d = 0; d < c->depth(); d++) ops_->mod_drop_inplace(p);
        ops_->multiply_plain_inplace(*c, p);
        ops_->rescale_inplace(*c);
        size_t n = 1;
        while (n < w.size()) n <<= 1;
        for (size_t step = 1; step < n; step <<= 1)
        {
            Ct r(*c);
            ops_->rotate_inplace(r, *galois_, static_cast<int>(step));
            ops_->add_inplace(*c, r);
            rotations_++;
        }
        return c;
    }


    // ct plus a plaintext vector, encoded at the ciphertext's own scale and depth. A plaintext addition changes only the
    // first component, so it is for arithmetic, not for masking: add_plain_slots masks.
    std::shared_ptr<Ct> add_plain(const std::shared_ptr<Ct>& a, const Reals& v)
    {
        heongpu::Plaintext<Scheme> p(context_);
        encoder_->encode(p, reals(v), a->scale());
        for (int d = 0; d < a->depth(); d++) ops_->mod_drop_inplace(p);
        auto c = std::make_shared<Ct>(*a);
        ops_->add_plain_inplace(*c, p);
        return c;
    }

    // ct times a plaintext vector, slot by slot, rescaled: one level used.
    std::shared_ptr<Ct> mul_plain(const std::shared_ptr<Ct>& a, const Reals& v)
    {
        heongpu::Plaintext<Scheme> p(context_);
        encoder_->encode(p, reals(v), scale_);
        for (int d = 0; d < a->depth(); d++) ops_->mod_drop_inplace(p);
        auto c = std::make_shared<Ct>(*a);
        ops_->multiply_plain_inplace(*c, p);
        ops_->rescale_inplace(*c);
        return c;
    }

    // GPU work is asynchronous: a timer read before this measures the launch, not the arithmetic.
    void synchronize() const { cudaDeviceSynchronize(); }

    long rotations() const { return rotations_; }

  private:
    heongpu::HEContext<Scheme> context_;
    double scale_;
    size_t ring_;
    long rotations_ = 0;
    std::unique_ptr<heongpu::HEKeyGenerator<Scheme>> keygen_;
    std::unique_ptr<heongpu::Secretkey<Scheme>> secret_;
    std::unique_ptr<heongpu::Publickey<Scheme>> public_;
    std::unique_ptr<heongpu::Galoiskey<Scheme>> galois_;
    std::unique_ptr<heongpu::HEEncoder<Scheme>> encoder_;
    std::unique_ptr<heongpu::HEEncryptor<Scheme>> encryptor_;
    std::unique_ptr<heongpu::HEDecryptor<Scheme>> decryptor_;
    std::unique_ptr<heongpu::HEArithmeticOperator<Scheme>> ops_;
};

PYBIND11_MODULE(qsh_heongpu, m)
{
    py::class_<Ct, std::shared_ptr<Ct>>(m, "Ciphertext")
        .def_property_readonly("scale", [](const Ct& c) { return c.scale(); })
        .def_property_readonly("depth", [](const Ct& c) { return c.depth(); });
    py::class_<Backend>(m, "Backend")
        .def(py::init<size_t, const std::vector<int>&, const std::vector<int>&, int>())
        .def_property_readonly("slot_count", &Backend::slot_count)
        .def_property_readonly("rotations", &Backend::rotations)
        .def("encrypt", &Backend::encrypt)
        .def("decrypt_slots", &Backend::decrypt_slots)
        .def("add", &Backend::add)
        .def("add_plain_slots", &Backend::add_plain_slots)
        .def("add_plain", &Backend::add_plain)
        .def("mul_plain", &Backend::mul_plain)
        .def("synchronize", &Backend::synchronize)
        .def("dot_plain", &Backend::dot_plain);
}
