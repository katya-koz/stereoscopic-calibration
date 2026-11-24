#pragma once

struct NormalizedCoords {
    double X = 0.0;
    double Y = 0.0;
};


struct NormalizedFixationCoordinates {
    NormalizedCoords Left = NormalizedCoords();
    NormalizedCoords Right = NormalizedCoords();
};