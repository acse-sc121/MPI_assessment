#define _USE_MATH_DEFINES

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <mpi.h>
#include <cstdlib>
#include <time.h>
#include <iomanip>
#include <chrono>

#define DO_TIMING

using namespace std;

int imax = 100, jmax = 100;
double t_max = 10;
double t, t_out = 0.0, dt_out = 0.04, dt;
double y_max = 10.0, x_max = 10.0, dx, dy;
double c = 1;
double** grid, ** new_grid, ** old_grid;
int id, p, tag_num = 1;

MPI_Request* request;

int rows, columns;     // the rows, columns in the domain


//Find the size (rows and columns) of subgrid
//From Exercise2_3
void find_dimensions(int p, int& rows, int& columns)
{
	int min_gap = p;
	int top = sqrt(p) + 1;
	for (int i = 1; i <= top; i++)
	{
		if (p % i == 0)
		{
			int gap = abs(p / i - i);

			if (gap < min_gap)
			{
				min_gap = gap;
				rows = i;
				columns = p / i;
			}
		}
	}
}

//Find the current row and column of id
void id_to_index(int id, int& id_row, int& id_column)
{
	id_column = id % columns;
	id_row = id / columns;
}

//Find the id from current row and column
int id_from_index(int id_row, int id_column)
{
	if (id_row >= rows || id_row < 0)
		return -1;
	if (id_column >= columns || id_column < 0)
		return -1;

	return id_row * columns + id_column;
}
//Initial condition - with one point
void disturbance(double** grid, double** old_grid, int subrows, int subcols, double dx, double dy, int id_row, int id_column)
{
	//sets half sinusoidal intitial disturbance - this is a bit brute force and it can be done more elegantly
	double r_splash = 1.0;
	double x_splash = 3.0;
	double y_splash = 3.0;

	for (int i = 1; i < subrows - 1; i++)
		for (int j = 1; j < subcols - 1; j++)
		{
			double x = dx * (i - 1 + ((int)(imax / rows) * id_row));
			double y = dy * (j - 1 + ((int)(jmax / columns) * id_column));

			double dist = sqrt(pow(x - x_splash, 2.0) + pow(y - y_splash, 2.0));

			//The points in the circle will be affected by the disturbance
			if (dist < r_splash)
			{
				double h = 5.0 * (cos(dist / r_splash * M_PI) + 1.0);

				grid[i][j] = h;
				old_grid[i][j] = h;
			}
		}


}

MPI_Datatype Datatype_left, Datatype_right, Datatype_top, Datatype_bottom, Datatype_left_recv, Datatype_right_recv, Datatype_top_recv, Datatype_bottom_recv;

//Create Datatype for sending
//which is 'Datatype_left, Datatype_right, Datatype_top, Datatype_bottom'
//From Exercise4_2
void create_types_send(double** data, int m, int n)
{
	vector<int> block_lengths;
	vector<MPI_Datatype> typelist;
	vector<MPI_Aint> addresses;
	MPI_Aint add_start;

	//left
	for (int i = 0; i < m; i++)
	{
		block_lengths.push_back(1);
		typelist.push_back(MPI_DOUBLE);
		MPI_Aint temp_address;
		MPI_Get_address(&data[i][0], &temp_address);
		addresses.push_back(temp_address);
	}
	MPI_Get_address(data, &add_start);
	for (int i = 0; i < m; i++) addresses[i] = addresses[i] - add_start;
	MPI_Type_create_struct(m, block_lengths.data(), addresses.data(), typelist.data(), &Datatype_left);
	MPI_Type_commit(&Datatype_left);

	//right
	block_lengths.resize(0);
	typelist.resize(0);
	addresses.resize(0);
	for (int i = 0; i < m; i++)
	{
		block_lengths.push_back(1);
		typelist.push_back(MPI_DOUBLE);
		MPI_Aint temp_address;
		MPI_Get_address(&data[i][n - 1], &temp_address);
		addresses.push_back(temp_address);
	}
	for (int i = 0; i < m; i++) addresses[i] = addresses[i] - add_start;
	MPI_Type_create_struct(m, block_lengths.data(), addresses.data(), typelist.data(), &Datatype_right);
	MPI_Type_commit(&Datatype_right);

	//top - only need one value
	int block_length = n;
	MPI_Datatype typeval = MPI_DOUBLE;
	MPI_Aint address;
	MPI_Get_address(data[0], &address);
	address = address - add_start;
	MPI_Type_create_struct(1, &block_length, &address, &typeval, &Datatype_top);
	MPI_Type_commit(&Datatype_top);

	//bottom - only need one value
	MPI_Get_address(data[m - 1], &address);
	address = address - add_start;
	MPI_Type_create_struct(1, &block_length, &address, &typeval, &Datatype_bottom);
	MPI_Type_commit(&Datatype_bottom);
}

//Create Datatype for receiving
//which is 'Datatype_left_recv, Datatype_right_recv, Datatype_top_recv, Datatype_bottom_recv'
//From Exercise4_2
void create_types_recv(double** data, int m, int n)
{
	vector<int> block_lengths;
	vector<MPI_Datatype> typelist;
	vector<MPI_Aint> addresses;
	MPI_Aint add_start;

	// left - need to exclude the corners of the ghost layer
	for (int i = 1; i < m - 1; i++)
	{
		block_lengths.push_back(1);
		typelist.push_back(MPI_DOUBLE);
		MPI_Aint temp_address;
		MPI_Get_address(&data[i][0], &temp_address);
		addresses.push_back(temp_address);
	}
	MPI_Get_address(data, &add_start);
	for (int i = 0; i < m - 2; i++) addresses[i] = addresses[i] - add_start;
	MPI_Type_create_struct(m - 2, block_lengths.data(), addresses.data(), typelist.data(), &Datatype_left_recv);
	MPI_Type_commit(&Datatype_left_recv);

	// right - need to exclude the corners of the ghost layer
	block_lengths.resize(0);
	typelist.resize(0);
	addresses.resize(0);
	for (int i = 1; i < m - 1; i++)
	{
		block_lengths.push_back(1);
		typelist.push_back(MPI_DOUBLE);
		MPI_Aint temp_address;
		MPI_Get_address(&data[i][n - 1], &temp_address);
		addresses.push_back(temp_address);
	}
	for (int i = 0; i < m - 2; i++) addresses[i] = addresses[i] - add_start;
	MPI_Type_create_struct(m - 2, block_lengths.data(), addresses.data(), typelist.data(), &Datatype_right_recv);
	MPI_Type_commit(&Datatype_right_recv);

	//top - only need one value
	int block_length = n - 2;
	MPI_Datatype typeval = MPI_DOUBLE;
	MPI_Aint address;
	//Begin with the first row, second column without corner
	MPI_Get_address(&data[0][1], &address);
	address = address - add_start;
	MPI_Type_create_struct(1, &block_length, &address, &typeval, &Datatype_top_recv);
	MPI_Type_commit(&Datatype_top_recv);

	//bottom - only need one value
	//Begin with the last row, second column without corner
	MPI_Get_address(&data[m - 1][1], &address);
	address = address - add_start;
	MPI_Type_create_struct(1, &block_length, &address, &typeval, &Datatype_bottom_recv);
	MPI_Type_commit(&Datatype_bottom_recv);
}

//Non-periodic
void do_communicate(double** send_array, double** recv_array, int subrows, int subcols, double** grid, int id_row, int id_column)
{
	int cnt = 0;
	//To know which edge doing communicate
	bool top_flag = false;
	bool bottom_flag = false;
	bool left_flag = false;
	bool right_flag = false;

	request = new MPI_Request[4 * 2];

	for (int i = -1; i <= 1; i++)
		for (int j = -1; j <= 1; j++)
		{
			int com_i = id_row + i;
			int com_j = id_column + j;

			//id of the grid communicate with
			int com_id = id_from_index(com_i, com_j);

			//Make sure it is a grid that can be communicated with
			if (com_id != id && com_id >= 0 && com_id < p)
			{
				//Communicate with top
				if (com_i == id_row - 1 && com_j == id_column)
				{
					MPI_Isend(send_array, 1, Datatype_top, com_id, tag_num, MPI_COMM_WORLD, &request[cnt]);
					cnt++;
					MPI_Irecv(recv_array, 1, Datatype_top_recv, com_id, tag_num, MPI_COMM_WORLD, &request[cnt]);
					cnt++;
					top_flag = true;

				}
				//Communicate with bottom
				if (com_i == id_row + 1 && com_j == id_column)
				{
					MPI_Isend(send_array, 1, Datatype_bottom, com_id, tag_num, MPI_COMM_WORLD, &request[cnt]);
					cnt++;
					MPI_Irecv(recv_array, 1, Datatype_bottom_recv, com_id, tag_num, MPI_COMM_WORLD, &request[cnt]);
					cnt++;
					bottom_flag = true;

				}
				//Communicate with left
				if (com_i == id_row && com_j == id_column - 1)
				{
					MPI_Isend(send_array, 1, Datatype_left, com_id, tag_num, MPI_COMM_WORLD, &request[cnt]);
					cnt++;
					MPI_Irecv(recv_array, 1, Datatype_left_recv, com_id, tag_num, MPI_COMM_WORLD, &request[cnt]);
					cnt++;
					left_flag = true;

				}
				//Communicate with right
				if (com_i == id_row && com_j == id_column + 1)
				{
					MPI_Isend(send_array, 1, Datatype_right, com_id, tag_num, MPI_COMM_WORLD, &request[cnt]);
					cnt++;
					MPI_Irecv(recv_array, 1, Datatype_right_recv, com_id, tag_num, MPI_COMM_WORLD, &request[cnt]);
					cnt++;
					right_flag = true;


				}
			}
		}
	MPI_Waitall(cnt, request, MPI_STATUS_IGNORE);

	//copy ghost layer into grid - exclude the corners
	if (top_flag)
	{
		for (int j = 1; j < subcols - 1; j++)
			grid[0][j] = recv_array[0][j];
	}

	if (bottom_flag)
	{
		for (int j = 1; j < subcols - 1; j++)
			grid[subrows - 1][j] = recv_array[subrows - 1][j];
	}

	if (left_flag)
	{
		for (int i = 1; i < subrows - 1; i++)
			grid[i][0] = recv_array[i][0];
	}

	if (right_flag)
	{
		for (int i = 1; i < subrows - 1; i++)
			grid[i][subcols - 1] = recv_array[i][subcols - 1];
	}
}

//Doing iteration for each grid - apply the discretised equation
void do_iteration(double** grid, double** new_grid, double** old_grid, int subrows, int subcols, int boundary, int id_row, int id_column, double dx, double dy, double dt)
{
	
	//Not the boundary process
	if (id_row != 0 && id_row != rows - 1 && id_column != 0 && id_column != columns - 1)
	{
		for (int i = 1; i < subrows - 1; i++)
			for (int j = 1; j < subcols - 1; j++)
			{
				new_grid[i][j] = pow(dt * c, 2.0) * ((grid[i + 1][j] - 2.0 * grid[i][j] + grid[i - 1][j]) / pow(dx, 2.0) + (grid[i][j + 1] - 2.0 * grid[i][j] + grid[i][j - 1]) / pow(dy, 2.0)) + 2.0 * grid[i][j] - old_grid[i][j];
			}
	}
	//Exclude ghost layer and bounday
	else if (id_row == 0) //Top processes
	{
		if (id_column == 0) //top-left corner process
		{
			//Begin with third row, third column
			//End at second last row, second last column
			for (int i = 2; i < subrows - 1; i++)
				for (int j = 2; j < subcols - 1; j++)
					new_grid[i][j] = pow(dt * c, 2.0) * ((grid[i + 1][j] - 2.0 * grid[i][j] + grid[i - 1][j]) / pow(dx, 2.0) + (grid[i][j + 1] - 2.0 * grid[i][j] + grid[i][j - 1]) / pow(dy, 2.0)) + 2.0 * grid[i][j] - old_grid[i][j];

		}
		else if (id_column == columns - 1) //top-right corner process
		{
			//Begin with third row, second column
			//End at sencond last row, third last column
			for (int i = 2; i < subrows - 1; i++)
				for (int j = 1; j < subcols - 2; j++)
					new_grid[i][j] = pow(dt * c, 2.0) * ((grid[i + 1][j] - 2.0 * grid[i][j] + grid[i - 1][j]) / pow(dx, 2.0) + (grid[i][j + 1] - 2.0 * grid[i][j] + grid[i][j - 1]) / pow(dy, 2.0)) + 2.0 * grid[i][j] - old_grid[i][j];
		}
		else //other top processes
		{
			//Begin with third row, second column
			//End at second last row, second last column
			for (int i = 2; i < subrows - 1; i++)
				for (int j = 1; j < subcols - 1; j++)
					new_grid[i][j] = pow(dt * c, 2.0) * ((grid[i + 1][j] - 2.0 * grid[i][j] + grid[i - 1][j]) / pow(dx, 2.0) + (grid[i][j + 1] - 2.0 * grid[i][j] + grid[i][j - 1]) / pow(dy, 2.0)) + 2.0 * grid[i][j] - old_grid[i][j];
		}
	}
	//Exclude ghost layer and bounday
	else if (id_row == rows - 1) //Bottom Processes
	{
		if (id_column == 0) //bottom-left corner process
		{
			//Begin with second row, third column
			//End at third last row, second last column
			for (int i = 1; i < subrows - 2; i++)
				for (int j = 2; j < subcols - 1; j++)
					new_grid[i][j] = pow(dt * c, 2.0) * ((grid[i + 1][j] - 2.0 * grid[i][j] + grid[i - 1][j]) / pow(dx, 2.0) + (grid[i][j + 1] - 2.0 * grid[i][j] + grid[i][j - 1]) / pow(dy, 2.0)) + 2.0 * grid[i][j] - old_grid[i][j];
		}
		else if (id_column == columns - 1) //bottom-right corner process
		{
			//Begin with second row, second column
			//End at third last row, third last column
			for (int i = 1; i < subrows - 2; i++)
				for (int j = 1; j < subcols - 2; j++)
					new_grid[i][j] = pow(dt * c, 2.0) * ((grid[i + 1][j] - 2.0 * grid[i][j] + grid[i - 1][j]) / pow(dx, 2.0) + (grid[i][j + 1] - 2.0 * grid[i][j] + grid[i][j - 1]) / pow(dy, 2.0)) + 2.0 * grid[i][j] - old_grid[i][j];
		}
		else //other bottom processes
		{
			//Begin with second row, second column
			//End at third last row, second last column
			for (int i = 1; i < subrows - 2; i++)
				for (int j = 1; j < subcols - 1; j++)
					new_grid[i][j] = pow(dt * c, 2.0) * ((grid[i + 1][j] - 2.0 * grid[i][j] + grid[i - 1][j]) / pow(dx, 2.0) + (grid[i][j + 1] - 2.0 * grid[i][j] + grid[i][j - 1]) / pow(dy, 2.0)) + 2.0 * grid[i][j] - old_grid[i][j];

		}
	}
	//Exclude ghost layer and bounday
	else if (id_column == 0 && id_row != 0 && id_row != rows - 1) //Left boundary but not corner
	{
		//Begin with second row, third column
		//End at second last row, second last column
		for (int i = 1; i < subrows - 1; i++)
			for (int j = 2; j < subcols - 1; j++)
				new_grid[i][j] = pow(dt * c, 2.0) * ((grid[i + 1][j] - 2.0 * grid[i][j] + grid[i - 1][j]) / pow(dx, 2.0) + (grid[i][j + 1] - 2.0 * grid[i][j] + grid[i][j - 1]) / pow(dy, 2.0)) + 2.0 * grid[i][j] - old_grid[i][j];
	}
	//Exclude ghost layer and bounday
	else if (id_column == columns - 1 && id_row != 0 && id_row != rows - 1) //Right boundary but not corner
	{
		//Begin with second row, second column
		//End at second last row, third last column
		for (int i = 1; i < subrows - 1; i++)
			for (int j = 1; j < subcols - 2; j++)
				new_grid[i][j] = pow(dt * c, 2.0) * ((grid[i + 1][j] - 2.0 * grid[i][j] + grid[i - 1][j]) / pow(dx, 2.0) + (grid[i][j + 1] - 2.0 * grid[i][j] + grid[i][j - 1]) / pow(dy, 2.0)) + 2.0 * grid[i][j] - old_grid[i][j];
	}

	//Implement boundary conditions - This is a Neumann boundary
	if (boundary == 0)
	{
		if (id_row == 0) //Top boundary
		{
			for (int j = 1; j < subcols - 1; j++)
				//Exclude the ghost layer
				new_grid[1][j] = new_grid[2][j];
		}

		if (id_row == rows - 1) //Bottom boundary
		{
			for (int j = 1; j < subcols - 1; j++)
				//Exclude the ghost layer
				new_grid[subrows - 2][j] = new_grid[subrows - 3][j];
		}

		if (id_column == 0) //Left boundary
		{
			for (int i = 1; i < subrows - 1; i++)
			{
				//Exclude the ghost layer
				new_grid[i][1] = new_grid[i][2];
			}
		}

		if (id_column == columns - 1) //Right boundary
			for (int i = 1; i < subrows - 1; i++)
			{
				//Exclude the ghost layer
				new_grid[i][subcols - 2] = new_grid[i][subcols - 3];
			}

	}
	//Implement boundary conditions - This is a Dirichlet boundary
	if (boundary == 1)
	{
		if (id_row == 0) //Top boundary
		{
			for (int j = 1; j < subcols - 1; j++)
				//Exclude the ghost layer
				new_grid[1][j] = 0;
		}

		if (id_row == rows - 1) //Bottom boundary
		{
			for (int j = 1; j < subcols - 1; j++)
				//Exclude the ghost layer
				new_grid[subrows - 2][j] = 0;
		}

		if (id_column == 0) //Left boundary
		{
			for (int i = 1; i < subrows - 1; i++)
			{
				//Exclude the ghost layer
				new_grid[i][1] = 0;
			}
		}

		if (id_column == columns - 1) //Right boundary
			for (int i = 1; i < subrows - 1; i++)
			{
				//Exclude the ghost layer
				new_grid[i][subcols - 2] = 0;
			}
	}

	t += dt;

	double** temp;
	double* _array = new double[subrows * subcols];
	temp = new double* [subrows];
	for (int i = 0; i < subrows; i++)
		temp[i] = &_array[i * subcols];

	//copy old grid into next new grid, grid into next old grid, new grid next into grid
	for (int i = 0; i < subrows - 1; i++)
		for (int j = 0; j < subcols - 1; j++)
		{
			temp[i][j] = old_grid[i][j];
			old_grid[i][j] = grid[i][j];
			grid[i][j] = new_grid[i][j];
			new_grid[i][j] = temp[i][j];
		}

	delete[] _array;
	delete[] temp;

}

//Write the output to files
void grid_to_file(int out, double** grid, int subrows, int subcols)
{
	stringstream fname;
	fstream f1;
	fname << "./out/id" << "_" << id << "_output_" << out << ".dat";
	f1.open(fname.str().c_str(), ios_base::out);
	//The size is for each grid
	for (int i = 1; i < subrows - 1; i++)
	{
		for (int j = 1; j < subcols - 1; j++) {

			f1 << grid[i][j] << "\t";
		}
		f1 << endl;
	}
	f1.close();
}

int main(int argc, char* argv[])
{
	int boundary = 1; //select boundary condition : 0 is Neumann, 1 is Dirichlet

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	int subrows, subcols;  // the rows, columns in the current grid
	int id_row, id_column; // the row, column of id

	find_dimensions(p, rows, columns);

	id_to_index(id, id_row, id_column);
	//Calculate subrows and subcols for each grid
	subrows = imax / rows + 2;
	subcols = jmax / columns + 2;
	int row_remain = imax % rows;
	int col_remain = jmax % columns;

	//If location of the current id is smaller than the total remainder, then add 1
	if (id / columns < row_remain) subrows += 1;
	if (id % columns < col_remain) subcols += 1;

	//Allocate the size of grid, old_grid, new_grid using 1D array mapped to 2D array(grids)
	double* array_1D_old = new double[subrows * subcols];
	double* array_1D_new = new double[subrows * subcols];
	double* array_1D = new double[subrows * subcols];

	old_grid = new double* [subrows];
	grid = new double* [subrows];
	new_grid = new double* [subrows];
	for (int i = 0; i < subrows; i++)
	{
		old_grid[i] = &array_1D_old[i * subcols];
		grid[i] = &array_1D[i * subcols];
		new_grid[i] = &array_1D_new[i * subcols];
	}

	//Initialize the grid, old_grid, new_grid with 0
	for (int i = 0; i < subrows; i++)
		for (int j = 0; j < subcols; j++)
		{
			old_grid[i][j] = 0;
			grid[i][j] = 0;
			new_grid[i][j] = 0;
		}

	dx = x_max / ((double)imax - 1);
	dy = y_max / ((double)imax - 1);

	t = 0.0;

	dt = 0.1 * min(dx, dy) / c;

	//Initial condition
	disturbance(grid, old_grid, subrows, subcols, dx, dy, id_row, id_column);

	int out_cnt = 0, it = 0;
	grid_to_file(out_cnt, grid, subrows, subcols);
	out_cnt++;
	t_out += dt_out;

	//Find the size of send array and recv array
	int m = subrows - 2;
	int n = subcols - 2;
	int m_recv = subrows;
	int n_recv = subcols;

	double** send_array, ** recv_array;

	//send_array need to store the grid without ghost layer
	send_array = new double* [m];
	for (int i = 0; i < m; i++)
		send_array[i] = new double[n];
	//Create datatype to send data
	create_types_send(send_array, m, n);

	//recv_array need to store the data with ghost layer
	recv_array = new double* [m_recv];
	for (int i = 0; i < m_recv; i++)
		recv_array[i] = new double[n_recv];
	//Create datatype to receive data
	create_types_recv(recv_array, m_recv, n_recv);

#ifdef DO_TIMING
	//The timing starts here
	auto start = chrono::high_resolution_clock::now();
#endif

	while (t < t_max)
	{
		//Initialize the data array with data of grid to send
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				//The grid has allocated extra 2 rows and columns, so the actual first number should plus 1
				send_array[i][j] = (double)grid[i + 1][j + 1];

		//Initialize the receive array
		for (int i = 0; i < m_recv; i++)
			for (int j = 0; j < n_recv; j++)
				recv_array[i][j] = 0;

		//start communicate
		do_communicate(send_array, recv_array, subrows, subcols, grid, id_row, id_column);

		//Iteration
		do_iteration(grid, new_grid, old_grid, subrows, subcols, boundary, id_row, id_column, dx, dy, dt);

		if (t_out <= t)
		{
			grid_to_file(out_cnt, grid, subrows, subcols);
			out_cnt++;
			t_out += dt_out;
		}

		it++;
	}

#ifdef DO_TIMING
	//Note that this should be done after a block in case process zero finishes quicker than the others
	//MPI_Waitall on process 0 is blocking for communications involving process 0, which, in this case is all the communications - Otherwise explicitly use MPI_Barrier
	auto finish = chrono::high_resolution_clock::now();
	if (id == 0)
	{
		std::chrono::duration<double> elapsed = finish - start;
		cout << setprecision(5);
		cout << "The code took " << elapsed.count() << "s to run" << endl;
	}
#endif

	//Delete all array to prevent stack leakage
	delete[] send_array;
	delete[] recv_array;
	delete[] old_grid;
	delete[] grid;
	delete[] new_grid;
	delete[] request;
	delete[] array_1D;
	delete[] array_1D_new;
	delete[] array_1D_old;


	MPI_Finalize();

}