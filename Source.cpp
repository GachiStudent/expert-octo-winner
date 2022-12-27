#include<iostream>
#include<omp.h>
#include<vector>
#include<chrono>
#include<ctime>


using my_matrix = std::vector<std::vector<double>>;
using my_vector = std::vector <double>;

void PrintMatrixObject(my_matrix& mat) {
	for (int i{ 0 }; i < mat.size(); ++i) {
		for (int j{ 0 }; j < mat.data()->size(); ++j) {
			std::cout << mat[i][j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

//Задание 1
void threads()
{
	int m{ 0 }, n{ 0 }, tid;
	std::cin >> m >> n;

	my_matrix matrix = my_matrix(m, my_vector(n, 0));
	for (auto& vec : matrix) {
		for (auto& elem : vec) {
			std::cin >> elem;
		}
	}

	#pragma omp parallel private(tid)
	{
		double Sum{ 0. };
		my_matrix sum{ my_matrix(1, my_vector(n, 0)) };
		tid = omp_get_thread_num();

		if (tid != 0) {
			for (int i{ 0 }; i < m; i += tid) {
				for (int j{ 0 }; j < n; ++j) {
					//Sum += matrix[i][j];
					sum[0][j] += matrix[i][j];
				}
			}
		}
		#pragma omp critical
		{
			//std::cout << "Номер нити: " << tid << ", Сумма: " << Sum << std::endl;
			std::cout << "Номер нити: " << tid << std::endl;
			PrintMatrixObject(sum);
		}
	}
}

//Задание 2
//https://www.cyberforum.ru/csharp-beginners/thread2910536.html

void FillMatrix()
{
    int k{ 0 }, n{ 0 };
    std::cin >> k >> n;

    my_matrix matrix = my_matrix(k, my_vector(n, 0));
	//PrintMatrixObject(matrix);

	int loops = k;
	if (loops > n) {
		loops = n;
	}
    loops = (loops + 1) / 2;

	auto start = std::chrono::system_clock::now();
	int i;
	#pragma omp parallel for private(i)
	for (i = 1; i <= loops; ++i) {
		int rowMin{ i - 1 };
		int columnMin {i - 1};

		int rowMax{ k - i };
		int columnMax{ n - i };

		for (int j{ columnMin }; j <= columnMax; ++j) {
			matrix[rowMin][j] = i;
			matrix[rowMax][j] = i;
		}

		for (int j{ rowMin }; j <= rowMax; ++j) {
			matrix[j][columnMin] = i;
			matrix[j][columnMax] = i;
		}
	}
	auto end = std::chrono::system_clock::now();

	PrintMatrixObject(matrix);
	std::cout <<"Время работы в миллисекундах: "<< (double)(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << "\n";
}

//Задание 3
//http://aco.ifmo.ru/el_books/numerical_methods/lectures/glava2_2.html
double f(double x) {
	return x * x * x;
}

# define M_PI           3.14159265358979323846
void integral_Simpsons_rule()
{
	int n{ 3 }; // Число разбиений n
	double a{ 0. }, b{ 1 }, integral{ 0. }; // Концы отрезка интегрирования a,b и сам интеграл
	double h{ (b - a) / n }; // Шаг интегрирования h

	std::cout << "Диапазон: a = " << a << ", b = " << b << std::endl;
	std::cout << "Функция: x^3\n";

	//Вычисляем интеграл по формуле Симпсона
	#pragma omp parallel for reduction(+:integral)
	for (int i{ 1 }; i <= n; ++i) {
		integral += f(a + h * (i - 0.5));
	}
	integral *= 4.;

	double TempSum{ 0. };
	#pragma omp parallel for reduction(+:TempSum)
	for (int i{ 1 }; i <= n - 1; ++i) {
		TempSum += f(a + h * i);
	}
	integral += TempSum * 2.;

	integral += f(a) + f(b);
	integral *= h / 6;

	std::cout << integral << "\n";
}

//Задание 4
// b*(A-C)* b_t, где: b - строка (1xN), A и C - матрицы (NxN)

namespace mat
{
	my_matrix Matrix_Transpose(const my_matrix& mat) {
		my_matrix transpose = my_matrix(mat.data()->size(), my_vector(mat.size(), 0));
		int i, j;
		#pragma omp parallel for private(i,j) collapse(2)
		for (i = 0; i < mat.size(); ++i) {
			for (j = 0; j < mat.data()->size(); ++j) {
				transpose[j][i] = mat[i][j];
			}
		}
		return transpose;
	}

	my_matrix Matrix_Subtraction(const my_matrix& mat1, const my_matrix& mat2) {
		my_matrix res = my_matrix(mat1.size(), my_vector(mat1.size(), 0));
		int i, j;
		#pragma omp parallel for private(i,j) collapse(2)
		for (i = 0; i < mat1.size(); ++i) {
			for (j = 0; j < mat1.size(); ++j) {
				res[i][j] = mat1[i][j] - mat2[i][j];
			}
		}
		return res;
	}

	my_matrix Matrix_Addition(const my_matrix& mat1, const my_matrix& mat2) {
		my_matrix res = my_matrix(mat1.size(), my_vector(mat1.size(), 0));
		int i, j;
		#pragma omp parallel for private(i,j) collapse(2)
		for (i = 0; i < mat1.size(); ++i) {
			for (j = 0; j < mat1.size(); ++j) {
				res[i][j] = mat1[i][j] + mat2[i][j];
			}
		}
		return res;
	}

	my_matrix Matrix_Multiplication(const my_matrix& mat1, const my_matrix& mat2) {
		my_matrix res = my_matrix(mat1.size(), my_vector(mat2.data()->size(), 0));
		int i, j, k;
		#pragma omp parallel for  private(i,j,k) collapse(3)
		for (i = 0; i < mat1.size(); ++i) {
			for (j = 0; j < mat2.data()->size(); ++j) {
				for (k = 0; k < mat2.size(); ++k) {
					res[i][j] += mat1[i][k] * mat2[k][j];
				}
			}
		}
		return res;
	}

	void Serial_Matrix_Multiplication()
	{
		int N;
		std::cout << "Размероность N: ";
		std::cin >> N;

		my_matrix b = my_matrix(1, my_vector(N, 0));
		my_matrix A = my_matrix(N, my_vector(N, 0));
		my_matrix C = my_matrix(N, my_vector(N, 0));
		my_matrix MainResult = my_matrix(1, my_vector(1, 0));

		srand(time(nullptr));
		int minRand = 10;
		int maxRand = 99;

		for (auto& vec : b) {
			for (auto& elem : vec) {
				//elem = minRand + rand() % (maxRand + 1 - minRand);
				elem = 1;
			}
		}

		for (auto& vec : A) {
			for (auto& elem : vec) {
				//elem = minRand + rand() % (maxRand + 1 - minRand);
				elem = 2;
			}
		}

		for (auto& vec : C) {
			for (auto& elem : vec) {
				//elem = minRand + rand() % (maxRand + 1 - minRand);
				elem = 1;
			}
		}

		std::cout << "Вектор b:\n";
		PrintMatrixObject(b);
		std::cout << "Матрица A:\n";
		PrintMatrixObject(A);
		std::cout << "Матрица C:\n";
		PrintMatrixObject(C);

		auto start = std::chrono::system_clock::now();
		my_matrix x1 = Matrix_Subtraction(A, C);
		my_matrix x2 = Matrix_Multiplication(b, x1);
		my_matrix x3 = Matrix_Transpose(b);
		MainResult = Matrix_Multiplication(x2, x3);
		auto end = std::chrono::system_clock::now();
		std::cout <<"Время работы алгоритма: "<< (double)(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " миллисекунд\n";

		std::cout << "Результат: ";
		PrintMatrixObject(MainResult);
	}
}

int main() {
	
	setlocale(LC_ALL, "ru");
	
	threads();

	//FillMatrix();

	//integral_Simpsons_rule();

	//mat::Serial_Matrix_Multiplication();

	system("pause");
	return 0;
}