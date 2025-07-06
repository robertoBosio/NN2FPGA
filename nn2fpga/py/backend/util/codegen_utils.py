from csnake import CodeWriter, Function, Variable
from backend.core.tensor_quant import TensorQuant

def get_stream_type(tensor_quant: TensorQuant, ch_par: int) -> str:
    """ Get the HLS stream type for a given tensor quantization and parallelization factor. """
    return f"hls::stream<{get_struct_type(tensor_quant, ch_par)}>"

def get_struct_type(tensor_quant: TensorQuant, ch_par: int) -> str:
    """ Get the HLS struct type for a given tensor quantization. """
    return f"std::array<{get_quant_type(tensor_quant)}, {ch_par}>"

def get_quant_type(tensor_quant: TensorQuant) -> str:
    """ Get the HLS type for a given tensor quantization. """
    return f"ap_{'' if tensor_quant.signed else 'u'}int<{tensor_quant.bitwidth}>"

def NewCodeWriter():
    """
    Create a new CodeWriter instance.
    """
    return CodeWriter()

class cpp_function(Function):
    def __init__(
        self,
        name: str,
        return_type: str = "void",
        qualifiers=None,
        arguments=None,
        templates=None,  # List[str], e.g. ["typename T", "int N"]
    ) -> None:
        super().__init__(name, return_type, qualifiers, arguments)
        self.templates = templates or []

    def _generate_template_prefix(self) -> str:
        if not self.templates:
            return ""
        return f"template <{', '.join(self.templates)}>"

    def generate_prototype(self, extern: bool = False) -> str:
        # Add template line before the standard prototype
        base = super().generate_prototype(extern)
        return self._generate_template_prefix() + base

    def generate_definition(self, indent="    "):
        # Add template line before the function definition
        writer = super().generate_definition(indent)
        if self.templates:
            tmpl_line = self._generate_template_prefix().strip()
            writer.lines.insert(0, tmpl_line)  # Insert before prototype
        return writer

    @property
    def prototype(self) -> str:
        return self.generate_prototype() + ";"

    @property
    def definition(self) -> str:
        return self.generate_definition().code

class cpp_variable(Variable):
    def __init__(self, name: str, primitive: str, *args, **kwargs):
        super().__init__(name, primitive, *args, **kwargs)

class cpp_object:

    def __init__(
        self,
        class_name: str,
        obj_name: str,
        template_args=None,  # List[Union[str, Tuple[str, str]]]
        constructor_args=None,  # List[Union[str, Tuple[str, str]]]
    ) -> None:

        self.class_name = class_name
        self.obj_name = obj_name
        self.template_args = template_args or []
        self.constructor_args = constructor_args or []

    def generate_declaration(self) -> str:
        """
        Generate a C++ object instantiation with optional template parameters and constructor arguments,
        supporting optional per-argument comments.

        Arguments:
            class_name:      Name of the class to instantiate
            obj_name:        Name of the object to declare
            template_args:   List of arguments or (arg, comment) tuples for template parameters
            constructor_args: List of arguments or (arg, comment) tuples for constructor arguments

        Returns:
            Formatted C++ declaration string.
        """

        cwr = CodeWriter()
        next_line = f"{self.class_name}"

        # Format template
        if self.template_args:
            next_line += " <"
            cwr.add_line(next_line)
            cwr.indent()
            for i, arg in enumerate(self.template_args):
                if i != len(self.template_args) - 1:
                    if isinstance(arg, tuple):
                        val, comment = arg
                        cwr.add_line(f"{val},  // {comment}")
                    else:
                        cwr.add_line(f"{arg},")
                else:
                    if isinstance(arg, tuple):
                        val, comment = arg
                        cwr.add_line(f"{val}  // {comment}")
                    else:
                        cwr.add_line(f"{arg}")
            cwr.dedent()
            next_line = f"> "

        next_line += f"{self.obj_name}"

        if self.constructor_args:
            next_line += f"("
            cwr.add_line(next_line)
            cwr.indent()
            for i, arg in enumerate(self.constructor_args):
                if i != len(self.constructor_args) - 1:
                    if isinstance(arg, tuple):
                        val, comment = arg
                        cwr.add_line(f"{val},  // {comment}")
                    else:
                        cwr.add_line(f"{arg},")
                else:
                    if isinstance(arg, tuple):
                        val, comment = arg
                        cwr.add_line(f"{val}  // {comment}")
                    else:
                        cwr.add_line(f"{arg}")
            cwr.dedent()
            next_line = f");"
        else:
            next_line += ";"

        cwr.add_line(next_line)
        return cwr.lines
