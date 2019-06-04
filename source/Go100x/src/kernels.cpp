
#include "Go100x/kernels.hpp"

Cell::Cell(int _max)
: m_max_size(_max)
{}

Grid::Grid(const dim_t& dims)
: m_dimensions(dims)
{}

void cpu::calculate(const Grid& grid)
{
    for(const auto& itr : grid)
    {
        std::cout << "cell: " << itr << std::endl;
    }
}
