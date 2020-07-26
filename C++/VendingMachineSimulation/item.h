#pragma once
#include <string>

class Item
{
	private: 
		const char *itemName;
		const char *itemType;
		double price;
	public:
		Item() {}
		Item(char n[], char t[], double p) { itemName = n; itemType = t; price = p; }
		double getPrice() { return price; }
		void printAttributes(int x, int y)
		{
			std::string name = itemName;
			std::cout	<< "(" << x << "," << y << ") " 
						<< "[" << name << " | " << price << "] ";
		}
};