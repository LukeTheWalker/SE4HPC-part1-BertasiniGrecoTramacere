#include "matrix_multiplication.h"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>

// ######################### Source code of multiplyMatrices in src/matrix_mult


TEST(MatrixMultiplicationPrerequisites, TestRowAWrong) {
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}
    };
    std::vector<std::vector<int>> B = {
        {7, 8},
        {9, 10},
        {11, 12}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C, 1, 3, 2);

    std::vector<std::vector<int>> expected = {
        {58, 64},
        {139, 154}
    };

    ASSERT_EQ(C, expected) << "Code does not check if passed row of A is correct";
}

TEST(MatrixMultiplicationPrerequisites, TestColAWrong) {
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}
    };
    std::vector<std::vector<int>> B = {
        {7, 8},
        {9, 10},
        {11, 12}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C, 2, 1, 2);

    std::vector<std::vector<int>> expected = {
        {58, 64},
        {139, 154}
    };

    ASSERT_EQ(C, expected) << "Code does not check if passed col of A is correct";
}

TEST(MatrixMultiplicationPrerequisites, TestColBWrong) {
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}
    };
    std::vector<std::vector<int>> B = {
        {7, 8},
        {9, 10},
        {11, 12}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C, 2, 1, 2);

    std::vector<std::vector<int>> expected = {
        {58, 64},
        {139, 154}
    };

    ASSERT_EQ(C, expected) << "Code does not check if passed row of B is correct";
}

TEST(MatrixMultiplicationPrerequisites, TestRowBWrong) {
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}
    };
    std::vector<std::vector<int>> B = {
        {7, 8},
        {9, 10},
        {11, 12},
        {13, 14}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C, 2, 4, 1);

    ASSERT_ANY_THROW() << "Code allows for wrong number of rowsA/colsB";
}

TEST(MatrixMultiplicationProperties, TestCommutative) {
    std::vector<std::vector<int>> A = {
        {1, 0},
        {1, -1}
    };
    std::vector<std::vector<int>> B = {
        {1, 2},
        {2, 4}
    };

    std::vector<std::vector<int>> C1(2, std::vector<int>(2, 0));
    std::vector<std::vector<int>> C2(2, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C1, 2, 2, 2);
    multiplyMatrices(B, A, C2, 2, 2, 2);

    ASSERT_NE(C1, C2) << "Matrix multiplication is commutative";
}

TEST(MatrixMultiplicationProperties, TestAssociativity) {
    std::vector<std::vector<int>> A = {
        {1, 0},
        {1, -1}
    };

    std::vector<std::vector<int>> B = {
        {1, 2},
        {2, 4}
    };

    std::vector<std::vector<int>> C = {
        { 7,  10},
        {-5, -9}
    };

    std::vector<std::vector<int>> AB(2, std::vector<int>(2, 0));
    multiplyMatrices(A, B, AB, 2, 2, 2);

    std::vector<std::vector<int>> AB_C(2, std::vector<int>(2, 0));
    multiplyMatrices(AB, C, AB_C, 2, 2, 2);

    std::vector<std::vector<int>> BC(2, std::vector<int>(2, 0));
    multiplyMatrices(B, C, BC, 2, 2, 2);

    std::vector<std::vector<int>> A_BC(2, std::vector<int>(2, 0));
    multiplyMatrices(A, BC, A_BC, 2, 2, 2);

    ASSERT_EQ(AB_C, A_BC) << "Matrix multiplication is not associative";
}

void sum_matrices(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, std::vector<std::vector<int>>& C) {
    // check if the matrices have the same size
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        throw std::invalid_argument("Matrices must have the same size");
    }
    // resize the result matrix
    C.resize(A.size(), std::vector<int>(A[0].size(), 0));
    for (int i = 0; i < A.size(); i++) {
        for (int j = 0; j < A[0].size(); j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

TEST(MatrixMultiplicationProperties, TestLeftDistributive) {
    std::vector<std::vector<int>> A = {
        {1, 0},
        {1, -1}
    };

    std::vector<std::vector<int>> B = {
        {1, 2},
        {2, 4}
    };

    std::vector<std::vector<int>> C = {
        { 7,  10},
        {-5, -9}
    };

    std::vector<std::vector<int>> BpC; 
    sum_matrices(B, C, BpC);

    std::vector<std::vector<int>> AB(2, std::vector<int>(2, 0));
    multiplyMatrices(A, B, AB, 2, 2, 2);

    std::vector<std::vector<int>> AC(2, std::vector<int>(2, 0));
    multiplyMatrices(A, C, AC, 2, 2, 2);

    std::vector<std::vector<int>> A_BpC(2, std::vector<int>(2, 0));
    multiplyMatrices(A, BpC, A_BpC, 2, 2, 2);

    std::vector<std::vector<int>> ABpAC;
    sum_matrices(AB, AC, ABpAC);

    ASSERT_EQ(A_BpC, ABpAC) << "Matrix multiplication is not left distributive";
}

TEST(MatrixMultiplicationProperties, TestRightDistributive) {
    std::vector<std::vector<int>> A = {
        {1, 0},
        {1, -1}
    };

    std::vector<std::vector<int>> B = {
        {1, 2},
        {2, 4}
    };

    std::vector<std::vector<int>> C = {
        { 7,  10},
        {-5, -9}
    };

    std::vector<std::vector<int>> BpC;
    sum_matrices(B, C, BpC);

    std::vector<std::vector<int>> BA(2, std::vector<int>(2, 0));
    multiplyMatrices(B, A, BA, 2, 2, 2);

    std::vector<std::vector<int>> CA(2, std::vector<int>(2, 0));
    multiplyMatrices(C, A, CA, 2, 2, 2);

    std::vector<std::vector<int>> BpC_A(2, std::vector<int>(2, 0));
    multiplyMatrices(BpC, A, BpC_A, 2, 2, 2);

    std::vector<std::vector<int>> BApCA(2, std::vector<int>(2, 0));
    sum_matrices(BA, CA, BApCA);

    ASSERT_EQ(BpC_A, BApCA) << "Matrix multiplication is not right distributive";
}

TEST(MatrixMultiplicationProperties, TesLeftIdentitySquare) {
    std::vector<std::vector<int>> A = {
        {1, 0},
        {1, -1}
    };

    std::vector<std::vector<int>> I = {
        {1, 0},
        {0, 1}
    };

    std::vector<std::vector<int>> IA(2, std::vector<int>(2, 0));
    multiplyMatrices(I, A, IA, 2, 2, 2);

    ASSERT_EQ(A, IA) << "Matrix multiplication does not have a left identity with square matrices";
}

TEST(MatrixMultiplicationProperties, TestRightIdentitySquare) {
    std::vector<std::vector<int>> A = {
        {1, 0},
        {1, -1}
    };

    std::vector<std::vector<int>> I = {
        {1, 0},
        {0, 1}
    };

    std::vector<std::vector<int>> AI(2, std::vector<int>(2, 0));
    multiplyMatrices(A, I, AI, 2, 2, 2);

    ASSERT_EQ(A, AI) << "Matrix multiplication does not have a right identity with square matrices";
}

TEST(MatrixMultiplicationProperties, TestLeftIdentityRect) {
    std::vector<std::vector<int>> A = {
        {1, 0},
        {1, -1}, 
        {4, 5}
    };

    std::vector<std::vector<int>> I = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };

    std::vector<std::vector<int>>IA(3, std::vector<int>(2, 0));
    multiplyMatrices(I, A, IA, 3, 2, 3);

    ASSERT_EQ(A, IA) << "Matrix multiplication does not have a left identity with rectangular matrices";
}

TEST(MatrixMultiplicationProperties, TestRightIdentityRect) {
    std::vector<std::vector<int>> A = {
        {1, 0},
        {1, -1}, 
        {4, 5}
    };

    std::vector<std::vector<int>> I = {
        {1, 0},
        {0, 1}
    };

    std::vector<std::vector<int>> AI(3, std::vector<int>(2, 0));
    multiplyMatrices(A, I, AI, 3, 2, 2);

    ASSERT_EQ(A, AI) << "Matrix multiplication does not have a left identity with square matrices";
}

TEST(MatrixMultiplicationProperties, TestProductByZero) {
    std::vector<std::vector<int>> A = {
        {1, 0},
        {1, -1}, 
        {4, 5}
    };

    std::vector<std::vector<int>> O = {
        {0, 0},
        {0, 0}
    };

    std::vector<std::vector<int>> ex_AO = {
          {0, 0},
          {0, 0},
          {0, 0}
      };

    std::vector<std::vector<int>> AO(3, std::vector<int>(2, 0));
    multiplyMatrices(A, O, AO, 3, 2, 2);

    ASSERT_EQ(AO, O) << "Matrix multiplication does not have a left identity with square matrices";
}

TEST(MatrixMultiplicationProperties, TestZeroByZero) {
    std::vector<std::vector<int>> O = {
        {0, 0},
        {0, 0}
    };

    std::vector<std::vector<int>> ex_OO = {
          {0, 0},
          {0, 0}
      };

    std::vector<std::vector<int>> OO(2, std::vector<int>(2, 0));
    multiplyMatrices(O, O, OO, 2, 2, 2);

    ASSERT_EQ(OO, ex_OO) << "zero by zero not working";
}


TEST(MatrixMultiplicationProperties, TestLargeNumber) {
    
    std::vector<std::vector<int>> A = {
        {1000000, 0},
        {1, -1}, 
        {4, 5}
    };
    std::vector<std::vector<int>> B = {
        {1, 3},
        {1, -1}, 
        {4, 5}

    };  
    std::vector<std::vector<int>> C(3, std::vector<int>(2, 0));
    multiplyMatrices(A, B, C, 3, 2, 2);

    std::vector<std::vector<int>> expected = {
        {1000000, 0},
        {0, 2},
        {21, 21}
    };

      ASSERT_EQ(C, expected) << "Matrix multiplication does not have a left identity with square matrices";
}

// TEST(MatrixMultiplicationProperties, Gigatest) {
    
//     std::vector<std::vector<int>> A(100, std::vector<int>(100, 0));
//     std::vector<std::vector<int>> B(100, std::vector<int>(100, 0));
//     std::vector<std::vector<int>> C(100, std::vector<int>(100, 0));

//     // Fill matrices with random integers
//     for (int i = 0; i < 100; i++) {
//       for (int j = 0; j < 100; j++) {
//         A[i][j] = rand() % 200 - 100 / 2;
//         B[i][j] = rand() % 200 - 100 / 2;
//       }
//     }

//     multiplyMatrices(A, B, C, 100, 100, 100);

//     std::vector<std::vector<int>> expected(100, std::vector<int>(100, 0));
//     for (int i = 0; i < 100; i++) {
//       for (int j = 0; j < 100; j++) {
//         for (int k = 0; k < 100; k++) {
//           expected[i][j] += A[i][k] * B[k][j];
//         }
//       }
//     }

//     ASSERT_EQ(C, expected) << "Matrix multiplication does not work for large matrices";
// }

// Error 1: Element-wise multiplication of ones detected!
TEST(SE4HPCTests, ElementWiseMultOfOnes) {
    std::vector<std::vector<int>> A = {
        {1}
    };
    std::vector<std::vector<int>> B = {
        {1}
    };

    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    multiplyMatrices(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected = {
        {1}
    };

    ASSERT_EQ(C, expected) << "Element-wise multiplication of matrices with ones does not work";
}

// Error 2: Matrix A contains the number 7!
TEST(SE4HPCTests, Mat1Contains7) {
    std::vector<std::vector<int>> A = {
        {7}
    };
  
    std::vector<std::vector<int>> B = {
        {1}
    };

    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    multiplyMatrices(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected = {
        {7}
    };

    ASSERT_EQ(C, expected) << "MatA contains the number 7!";
}

// Error 3: Matrix A contains a negative number!
TEST(SE4HPCTests, Mat1ContainsNegNumber) {
    std::vector<std::vector<int>> A = {
        {-1}
    };

    std::vector<std::vector<int>> B = {
        {1}
    };

    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    multiplyMatrices(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected = {
        {-1}
    };

    ASSERT_EQ(C, expected) << "MatA contains a negative number!";
}

// Error 4: Matrix B contains the number 3!
TEST(SE4HPCTests, Mat2Contains3) {
    std::vector<std::vector<int>> A = {
        {1}
    };

    std::vector<std::vector<int>> B = {
        {3}
    };

    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    multiplyMatrices(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected = {
        {3}
    };

    ASSERT_EQ(C, expected) << "MatB contains the number 3!";
}

// Error 5: Matrix B contains a negative number!
TEST(SE4HPCTests, Mat2ContainsNegNumber) {
    std::vector<std::vector<int>> A = {
        {1}
    };

    std::vector<std::vector<int>> B = {
        {-1}
    };

    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    multiplyMatrices(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected = {
        {-1}
    };

    ASSERT_EQ(C, expected) << "MatB contains a negative number!";
}

// Error 6: Result matrix contains a number bigger than 100!
TEST(SE4HPCTests, ResultMatrixContainsBigNumber) {
    std::vector<std::vector<int>> A = {
        {101}
    };

    std::vector<std::vector<int>> B = {
        {1}
    };

    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    multiplyMatrices(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected = {
        {101}
    };

    ASSERT_EQ(C, expected) << "Result matrix contains a number bigger than 100!";
}

// Error 7: Result matrix contains a number between 11 and 20!
TEST(SE4HPCTests, ResultMatrixContainsNumberBetween11And20) {
    std::vector<std::vector<int>> A = {
        {12}
    };

    std::vector<std::vector<int>> B = {
        {1}
    };

    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    multiplyMatrices(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected = {
        {12}
    };

    ASSERT_EQ(C, expected) << "Result matrix contains a number between 11 and 20!";
}

// Error 8: Result matrix contains zero!
TEST(SE4HPCTests, ResultMatrixContainsZero) {
    std::vector<std::vector<int>> A = {
        {0}
    };

    std::vector<std::vector<int>> B = {
        {0}
    };

    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    multiplyMatrices(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected = {
        {0}
    };

    ASSERT_EQ(C, expected) << "Result matrix contains zero!";
}

// Error 9: Result matrix contains the number 99!
TEST(SE4HPCTests, ResultMatrixContains99) {
    std::vector<std::vector<int>> A = {
        {99}
    };

    std::vector<std::vector<int>> B = {
        {1}
    };

    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    multiplyMatrices(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected = {
        {99}
    };

    ASSERT_EQ(C, expected) << "Result matrix contains the number 99!";
}

// Error 10: A row in matrix A contains more than one '1'!
TEST(SE4HPCTests, RowInMatAContainsMoreThanOne1) {
    std::vector<std::vector<int>> A = {
        {1, 1}
    };

    std::vector<std::vector<int>> B = {
        {1},
        {1}
    };

    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    multiplyMatrices(A, B, C, 1, 2, 1);

    std::vector<std::vector<int>> expected = {
        {2}
    };

    ASSERT_EQ(C, expected) << "A row in matrix A contains more than one '1'!";
}

// Error 11: Every row in matrix B contains at least one '0'!
TEST(SE4HPCTests, RowInMatBContainsAtLeastOne0) {
    std::vector<std::vector<int>> A = {
        {1}
    };

    std::vector<std::vector<int>> B = {
        {0}
    };

    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    multiplyMatrices(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected = {
        {0}
    };

    ASSERT_EQ(C, expected) << "Every row in matrix B contains at least one '0'!";
}

// Error 12: The number of rows in A is equal to the number of columns in B!
TEST(SE4HPCTests, NumOfRowsInAEqualToNumOfColsInB) {
    std::vector<std::vector<int>> A = {
        {1, 2},
        {3, 4}
    };

    std::vector<std::vector<int>> B = {
        {1, 2},
        {3, 4}
    };

    std::vector<std::vector<int>> C(2, std::vector<int>(1, 0));
    multiplyMatrices(A, B, C, 2, 2, 2);

    std::vector<std::vector<int>> expected = {
      {7, 10},
      {15, 22}
    };
    
    ASSERT_EQ(C, expected) << "The number of rows in A is equal to the number of columns in B!";
}

// Error 13: The first element of matrix A is equal to the first element of matrix B!
TEST(SE4HPCTests, FirstElementOfMatAEqualToFirstElementOfMatB) {
    std::vector<std::vector<int>> A = {
        {1}
    };

    std::vector<std::vector<int>> B = {
        {1}
    };

    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    multiplyMatrices(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected = {
        {1}
    };

    ASSERT_EQ(C, expected) << "The first element of matrix A is equal to the first element of matrix B!";
}

// Error 14: The result matrix C has an even number of rows!
TEST(SE4HPCTests, ResultMatrixHasEvenNumOfRows) {
    std::vector<std::vector<int>> A = {
        {1},
        {2}
    };

    std::vector<std::vector<int>> B = {
        {1}
    };

    std::vector<std::vector<int>> C(2, std::vector<int>(1, 0));
    multiplyMatrices(A, B, C, 2, 1, 1);

    std::vector<std::vector<int>> expected = {
        {1},
        {2}
    };

    ASSERT_EQ(C, expected) << "The result matrix C has an even number of rows!";
}

// Error 15: A row in matrix A is filled entirely with 5s!
TEST(SE4HPCTests, RowInMatAIsFilledWith5s) {
    std::vector<std::vector<int>> A = {
        {5, 5}
    };

    std::vector<std::vector<int>> B = {
        {1},
        {1}
    };

    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    multiplyMatrices(A, B, C, 1, 2, 1);

    std::vector<std::vector<int>> expected = {
        {10}
    };

    ASSERT_EQ(C, expected) << "A row in matrix A is filled entirely with 5s!";
}

// Error 16: Matrix B contains the number 6!
TEST(SE4HPCTests, MatBContains6) {
    std::vector<std::vector<int>> A = {
        {1}
    };

    std::vector<std::vector<int>> B = {
        {6}
    };

    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    multiplyMatrices(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected = {
        {6}
    };

    ASSERT_EQ(C, expected) << "MatB contains the number 6!";
}

// Error 17: Result matrix C contains the number 17!
TEST(SE4HPCTests, ResultMatrixContains17) {
    std::vector<std::vector<int>> A = {
        {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15}, {16}, {17}, {18}, {19}, {20}, {21}, {22}, {23}, {24}, {25}, {26}, {27}, {28}, {29}, {30}, {31}, {32}, {33}, {34}, {35}, {36}, {37}, {38}, {39}, {40}, {41}, {42}, {43}, {44}, {45}, {46}, {47}, {48}, {49}, {50}, {51}, {52}, {53}, {54}, {55}, {56}, {57}, {58}, {59}, {60}, {61}, {62}, {63}, {64}, {65}, {66}, {67}, {68}, {69}, {70}, {71}, {72}, {73}, {74}, {75}, {76}, {77}, {78}, {79}, {80}, {81}, {82}, {83}, {84}, {85}, {86}, {87}, {88}, {89}, {90}, {91}, {92}, {93}, {94}, {95}, {96}, {97}, {98}, {99}, {100}
    };

    std::vector<std::vector<int>> B = {
        {1}
    };

    std::vector<std::vector<int>> C(100, std::vector<int>(1, 0));
    multiplyMatrices(A, B, C, 100, 1, 1);

    std::vector<std::vector<int>> expected = {
      {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15}, {16}, {17}, {18}, {19}, {20}, {21}, {22}, {23}, {24}, {25}, {26}, {27}, {28}, {29}, {30}, {31}, {32}, {33}, {34}, {35}, {36}, {37}, {38}, {39}, {40}, {41}, {42}, {43}, {44}, {45}, {46}, {47}, {48}, {49}, {50}, {51}, {52}, {53}, {54}, {55}, {56}, {57}, {58}, {59}, {60}, {61}, {62}, {63}, {64}, {65}, {66}, {67}, {68}, {69}, {70}, {71}, {72}, {73}, {74}, {75}, {76}, {77}, {78}, {79}, {80}, {81}, {82}, {83}, {84}, {85}, {86}, {87}, {88}, {89}, {90}, {91}, {92}, {93}, {94}, {95}, {96}, {97}, {98}, {99}, {100}
    };

    ASSERT_EQ(C, expected) << "Matrix contains 17 (?)";
}


// Error 18: Matrix A is a square matrix!
TEST(SE4HPCTests, MatAIsSquare) {
    std::vector<std::vector<int>> A = {
        {1, 1},
        {1, 1}
    };

    std::vector<std::vector<int>> B = {
        {0, 0},
        {0, 0}
    };

    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));
    multiplyMatrices(A, B, C, 2, 2, 2);

    std::vector<std::vector<int>> expected = {
      {0, 0},
      {0, 0}
    };
    
    ASSERT_EQ(C, expected) << "Matrix A is a square matrix!";
}

// Error 19: Every row in matrix A contains the number 8!
TEST(SE4HPCTests, RowInMatAContains8) {
    std::vector<std::vector<int>> A = {
        {8, 8}
    };

    std::vector<std::vector<int>> B = {
        {1},
        {1}
    };

    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    multiplyMatrices(A, B, C, 1, 2, 1);

    std::vector<std::vector<int>> expected = {
        {16}
    };

    ASSERT_EQ(C, expected) << "Every row in matrix A contains the number 8!";
}

// Error 20: Number of columns in matrix A is odd!
TEST(SE4HPCTests, NumOfColsInMatAIsOdd) {
    std::vector<std::vector<int>> A = {
        {1, 2, 3}
    };

    std::vector<std::vector<int>> B = {
        {1},
        {1},
        {1}
    };

    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    multiplyMatrices(A, B, C, 1, 3, 1);

    std::vector<std::vector<int>> expected = {
        {6}
    };

    ASSERT_EQ(C, expected) << "Number of columns in matrix A is odd!";
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}