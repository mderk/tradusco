translation = open(f"{__package__}/translation.txt", "r").read()

output_format = open(f"{__package__}/output_format.txt", "r").read()


__all__ = [
    "translation",
    "output_format",
]
