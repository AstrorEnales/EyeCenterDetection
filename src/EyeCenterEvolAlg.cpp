//#include "Utils.h"
#include "Gradient.h"
#include "EyeCenterEvolAlg.h"
#include <stdio.h>
#include <random>

using namespace cv;

namespace EyeCenterEvolAlg {
  double fitnessEvolAlg(Mat& image, Mat& grad_x, Mat& grad_y, int cx, int cy) {
    double fitness = 0, length, dot;
    Point2f d, g;
    for(int y = 0; y < image.rows; y++) {
      for(int x = 0; x < image.cols; x++) {
        // Normalized distance vector
        d = Point2f(x - cx, y - cy);
        length = sqrt(d.x * d.x + d.y * d.y);
        if(length > 0) {
          d /= length;
        }
        // Normalized gradient vector
        g = Point2f(grad_x.at<float>(y, x), grad_y.at<float>(y, x));
        dot = d.dot(g);
        fitness += dot * dot;
      }
    }
    int N = image.cols * image.rows;
    return fitness / N;
  }

  struct Individual { 
    int x;
    int y;
    float fit;
  };
  
  const int num_generations = 200;
  const int num_individuals = 20;
  const int mutation_rate = 15;
  const int crossover_rate = 60;
  const int num_elitism = 0; // # fittest always survive
  std::random_device rd;     // only used once to initialise (seed) engine
  std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
  std::uniform_int_distribution<int> percent_dist(0, 99);
  std::uniform_int_distribution<int> individuals_dist(0, num_individuals - 1);

  int binaryTournament(std::vector<Individual> *individuals) {
    int i1 = individuals_dist(rng);
    int i2 = individuals_dist(rng);
    return individuals->at(i1).fit > individuals->at(i2).fit ? i1 : i2;
  }

  int findEyeCenters(Mat& image, Point*& centers, bool silentMode) {
    struct by_fitness { 
        bool operator()(Individual const &a, Individual const &b) { 
            return a.fit > b.fit;
        }
    };
  
    Mat grey(image.size(), CV_8UC1);
    cvtColor(image, grey, CV_RGB2GRAY);

    Mat grad_x, grad_y;
    calculateGradients(Four_Neighbor, grey, grad_x, grad_y);


    std::vector<Individual> *individuals = new std::vector<Individual>;

    //srand(1337);
    
    std::uniform_int_distribution<int> x_dist(0, image.cols - 1);
    std::uniform_int_distribution<int> y_dist(0, image.rows - 1);
    std::cout << "Creating " << num_individuals << " individuals..." << std::endl;

    for(int i = 0; i < num_individuals; i++) {
      Individual ind;
      ind.x = x_dist(rng);
      ind.y = y_dist(rng);
      ind.fit = fitnessEvolAlg(image, grad_x, grad_y, ind.x, ind.y);
      individuals->push_back(ind);
    }

    std::sort(individuals->begin(), individuals->end(), by_fitness());
    
    std::cout << "Start evolution..." << std::endl;
    for(int gen = 1; gen <= num_generations; gen++) {
      if(gen % 10 == 0)
        std::cout << "GEN " << gen << "/" << num_generations << " (# ind: " << individuals->size() << ")" << std::endl;
      for(int i = 0; i < individuals->size(); i++) {
        //std::cout << "\tIND " << i << ": " << individuals->at(i).loc << " = " << individuals->at(i).fit << std::endl;
      }
      std::vector<Individual> *next_gen = new std::vector<Individual>;

      // # number of elites always survive
      for(int i = 0; i < num_elitism; i++) {
        next_gen->push_back(individuals->at(i));
      }
    
      while(next_gen->size() < num_individuals) {
        
        int p1 = binaryTournament(individuals);
        int p2 = binaryTournament(individuals);
        Individual parent1 = individuals->at(p1);
        Individual parent2 = individuals->at(02);
        Individual child;
        child.x = parent1.x;
        child.y = parent1.y;
        // Point mutate at random x or y with random offset in range [-half, half] of either width or height
        if(percent_dist(rng) < mutation_rate) {
          if(percent_dist(rng) < 50) {
            child.x = child.x + x_dist(rng) - (image.cols / 2);
            child.x = (child.x + image.cols) % image.cols;
          } else {
            child.y = child.y + y_dist(rng) - (image.rows / 2);
            child.y = (child.y + image.rows) % image.rows;
          }
        }
        // Crossover parent1 with parent2. Since we only have two positions, always swap y.
        if(percent_dist(rng) < crossover_rate) {
          child.y = parent2.y;
        }
        child.fit = fitnessEvolAlg(image, grad_x, grad_y, child.x, child.y);
        next_gen->push_back(child);
      }

      std::sort(next_gen->begin(), next_gen->end(), by_fitness());
      individuals = next_gen;
    }

    centers = new Point[1];
    centers[0] = Point(individuals->at(0).x, individuals->at(0).y);
    return 1;
  }
}