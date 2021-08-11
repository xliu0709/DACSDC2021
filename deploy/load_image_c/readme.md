## Use c code to load the images
- command:  g++ -shared -O3 load_image_new2.cpp -o load_image_new2.so -fPIC `pkg-config opencv --cflags --libs` -lpthread
