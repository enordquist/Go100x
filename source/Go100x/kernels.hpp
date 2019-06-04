
#pragma once

#include "Go100x/macros.hpp"
#include <iostream>
#include <tuple>
#include <vector>

using dim_t = std::tuple<int, int, int>;

//======================================================================================//
// dummy cell class
//
class Cell
{
public:
    Cell(int _max);

    friend std::ostream& operator<<(std::ostream& os, const Cell& c)
    {
        os << "max = " << c.m_max_size << ", current = " << c.m_current_size;
        return os;
    }

protected:
    int              m_current_size = 0;
    int              m_max_size     = 50;
    std::vector<int> m_indices;
};

//======================================================================================//
// dummy grid class
//
class Grid
{
public:
    Grid(const dim_t&);

    auto begin() { return m_cells.begin(); }
    auto begin() const { return m_cells.begin(); }
    auto end() { return m_cells.end(); }
    auto end() const { return m_cells.end(); }

protected:
    dim_t             m_dimensions;
    std::vector<Cell> m_cells;
};

//======================================================================================//
// dummy cpu function that launches calcuations
//
namespace cpu
{
static void calculate(const Grid&);
}

//======================================================================================//
// dummy cuda kernel
//
namespace gpu
{
__global__ static void calculate(int size, int* indices);
    
__global__ static void fillNeighbors(std::vector<floats> x_arr, std::vector<floats> y_arr, std::vector<floats> z_arr);

}
