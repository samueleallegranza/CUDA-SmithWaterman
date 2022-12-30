# GPU optimization of Smith–Waterman algorithm

Multiple solutions have been developed, with the intent of trying to test different approaches.
Here is a summary of the different versions that can be found in this repository. Please refer to the project report (`report.pdf`) for the explaination of solutions.

- `sw_cuda_v2` : Solution that implements the *2-Threads* approach.
- `sw_cuda_v3` : Solution that implements the *L-Threads* approach. 
- `sw_cuda_v2` : Solution that implements the *L-Threads* approach, with the difference of spanning the matrices into vectors.

Please note that `sw_cuda_v0` and `sw_cuda_v1` should not be taken in consideration. 

To run the solutions, compile them with `nvcc -O3 sw_cuda_vX`, where `X` stands for the solution's version number. To run the compiled program, use `./a.out`

The `dev` folder contains solutions currently under development. Not being able to compile them is expected.
The `report_src` folder contains LaTeX source files for the report, which can be consulted by opening `report.pdf`.

Thanks for your time!

*Developed by Samuele Allegranza*