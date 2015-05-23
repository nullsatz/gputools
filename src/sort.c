#include<stdio.h>
#include<R.h>

#include"cuseful.h"
#include"sort.h"

void copyVect(int n, double * from, int incFrom, double * to, int incTo)
{
	for(int i = 0; i < n; i++)
		to[i*incTo] = from[i*incFrom];
}

//Swap rows in a col major array
void swapRows(int rows, int cols, double * array, int rowA, int rowB)
{
	double * tempRow  = Calloc(cols, double);

	copyVect(cols, array+rowA, rows, tempRow, 1);
	copyVect(cols, array+rowB, rows, array+rowA, rows);
	copyVect(cols, tempRow, 1, array+rowB, rows);
	Free(tempRow);
}

//Find the index of the Median of the elements
//of array that occur at every "shift" positions.
int findMedianIndex(int rows, int cols, int colToSortOn, double * array, 
	int left, int right, int shift)
{
	int 
		i, 
		groups = (right - left)/shift + 1, 
		k = left + groups/2*shift;

	double 
		* colToSort = array+colToSortOn*rows;

	for(i = left; i <= k; i += shift) {

		int 
			minRow = i; 
		double
			minValue = colToSort[minRow];

		for(int j = i; j <= right; j +=shift) {
			if(colToSort[j] < minValue) {
				minRow = j;
				minValue = colToSort[minRow];
			}
		}
		swapRows(rows, cols, array, i, minRow);
	}
	return k;
}
 
//Computes the median of each group of 5 elements and stores
//it as the first element of the group. Recursively does this
//till there is only one group and hence only one Median
double findMedianOfMedians(int rows, int cols, int colToSortOn, double * array, 
	int left, int right)
{
	double * colToSort = array+colToSortOn*rows;
	if(left == right)
		return colToSort[left];
 
	int i, shift = 1;
	while(shift <= (right - left)) {
		for(i = left; i <= right; i+=shift*5) {
			int endIndex = (i + shift*5 - 1 < right) ? i + shift*5 - 1 : right;

			int medianIndex = findMedianIndex(rows, cols, colToSortOn, array, 
				i, endIndex, shift);

			swapRows(rows, cols, array, i, medianIndex);
		}
		shift *= 5;
	}
	return colToSort[left];
}

//Partition the array into two halves and return the
//index about which the array is partitioned
int partition(int rows, int cols, int colToSortOn, double * array, 
	int left, int right)
{
	//Makes the leftmost element a good pivot,
	//specifically the median of medians
	findMedianOfMedians(rows, cols, colToSortOn, array, left, right);

	int 
		pivotIndex = left, index = left, 
		i;
	double 
		* colToSort = array+colToSortOn*rows,
		pivotValue = colToSort[pivotIndex];
 
	swapRows(rows, cols, array, pivotIndex, right);

	for(i = left; i < right; i++) {
		if(colToSort[i] < pivotValue) {
			swapRows(rows, cols, array, i, index);
			index += 1;
		}
	}
	swapRows(rows, cols, array, right, index);
	return index;
}
 
//Quicksort the array
void quicksort(int rows, int cols, int colToSortOn, double * array, 
	int left, int right)
{
	if(left >= right)
		return;
 
	int index = partition(rows, cols, colToSortOn, array, left, right);
	quicksort(rows, cols, colToSortOn, array, left, index - 1);
	quicksort(rows, cols, colToSortOn, array, index + 1, right);
}
