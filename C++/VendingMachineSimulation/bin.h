#pragma once
#include "item.h"
class Bin
{
	private: 
		Item binItem;
		bool isEmpty;
	public:
		Bin() { isEmpty = false; }
		Bin(Item item, int x, int y) { binItem = item; xcoord = x; ycoord = y; isEmpty = false; }
		int xcoord;
		int ycoord;
		Item getItem() { isEmpty = true; return binItem; }
		bool binIsEmpty() { return isEmpty; }
		void setEmptyFlag(bool flag) { isEmpty = flag; }
		void printItemsFromBin() 
		{ 
			if (!isEmpty) binItem.printAttributes(xcoord, ycoord); 
		}
};