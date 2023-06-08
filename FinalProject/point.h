#ifndef POINT_H
#define POINT_H

typedef struct {
    double x1;
    double x2;
    double a;
    double b;
    int id;
} Point;

#endif


double calculateDistance(Point p1, Point p2, int t);