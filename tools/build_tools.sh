mkdir -p build
cd build

# build capsule_infer
python3 -m cython --embed -3 -o capsule_infer.c ../capsule_infer/capsule_infer.py
gcc -Os -I /usr/include/python3.8 -o capsule_infer capsule_infer.c -lpython3.8 -lpthread -lm -lutil -ldl -Wl,--gc-sections -Wl,--strip-all

# build capsule_infer.cpython-38-x86_64-linux-gnu.so
cythonize -i ../capsule_infer/capsule_infer.py
strip ../capsule_infer/capsule_infer.cpython-38-x86_64-linux-gnu.so
cp ../capsule_infer/capsule_infer.cpython-38-x86_64-linux-gnu.so .

# build capsule_classifier_accuracy
python3 -m cython --embed -3 -o classifier_accuracy.c ../capsule_classifier_accuracy/classifier_accuracy.py
gcc -Os -I /usr/include/python3.8 -o capsule_classifier_accuracy classifier_accuracy.c ../capsule_infer/capsule_infer.cpython-38-x86_64-linux-gnu.so -lpython3.8 -lpthread -lm -lutil -ldl -Wl,--gc-sections -Wl,--strip-all

# remove all build output
#
# rm capsule_infer/capsule_infer.c capsule_infer/capsule_infer.cpython-38-x86_64-linux-gnu.so
# rm -rf build capsule_infer/build
