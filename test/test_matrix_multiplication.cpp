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

TEST(MatrixMultiplicationProperties, TestLargeNumber) {
    
    std::vector<std::vector<int>> A = {
        {1000000, 0},
        {1, -1}, 
        {4, 5}
    };
    std::vector<std::vector<int>> B = {
        {1, 0},
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

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}