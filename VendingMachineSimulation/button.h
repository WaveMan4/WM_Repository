#pragma once
class Button
{
	private:
		bool isEmpty;
		int xcoord;
		int ycoord;
	public: 
		Button() { isEmpty = false; xcoord = 0; ycoord = 0; }
		Button(int x, int y) { xcoord = x; ycoord = y; isEmpty = false; }
		void setxcoord(int val) { xcoord = val; }
		void setycoord(int val) { ycoord = val; }
		int getxcoord() { return xcoord; }
		int getycoord() { return ycoord; }
};