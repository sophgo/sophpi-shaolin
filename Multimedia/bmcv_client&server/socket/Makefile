CXX = g++
CXXFLAGS = -Wall -g -std=c++11
LDLIBS = -lpthread

all : server client
.PHONY : all

client :client.o
	$(CXX) $(CXXFLAGS) -o client  client.o

server : server.o
	$(CXX) $(CXXFLAGS) -o server   server.o  $(LDLIBS)

server.o : server.cpp
	$(CXX) $(CXXFLAGS) -c server.cpp 

client.o : client.cpp
	$(CXX) $(CXXFLAGS) -c client.cpp 

