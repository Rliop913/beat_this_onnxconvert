from conan import ConanFile
from conan.tools.build import check_min_cppstd


class BeatThisPreprocessConan(ConanFile):
    required_conan_version = ">=2.21.0"
    name = "beat_this_preprocess_cpp"
    version = "0.1"

    settings = "os", "arch", "compiler", "build_type"
    generators = "CMakeDeps", "CMakeToolchain"
    default_options = {
        "onnx/*:disable_static_registration": True,
    }

    def requirements(self):
        self.requires("onnxruntime/1.24.4")

    def validate(self):
        check_min_cppstd(self, 20)
