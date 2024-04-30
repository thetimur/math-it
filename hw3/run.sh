python main.py --mode compress --method numpy --compression 4 --in_file images/sample_1920×1280.bmp --out_file compressed_numpy.pkl

python main.py --mode compress --method simple --compression 4 --in_file images/sample_1920×1280.bmp --out_file compressed_simple.pkl

python main.py --mode compress --method advanced --compression 4 --in_file images/sample_1920×1280.bmp --out_file compressed_advanced.pkl

python main.py --mode decompress --in_file compressed_numpy.pkl --out_file decompressed_numpy.bmp

python main.py --mode decompress --in_file compressed_simple.pkl --out_file decompressed_simple.bmp

python main.py --mode decompress --in_file compressed_advanced.pkl --out_file decompressed_advanced.bmp
