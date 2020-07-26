package kepnang.gilles.jewelmatcher;

import android.graphics.Rect;

/**
 * Created by scott on 5/8/2016.
 */
public class Thing {
	//"Thing" identifies as a type of shape
	public enum Type {
		Square, Circle, Diamond, GemStone, NoShape;
	}
	private Type type;
	private Rect bounds;
	private int row;
	private int column;

	public Thing(int column, int row, Type type, Rect bounds) {
		this.column = column;
		this.row = row;
		this.type = type;
		this.bounds = bounds;
	}

	public int getColumn() {return column;}
	public int getRow() {return row;}
	public void setColumn(int column) {this.column = column; }
	public void setRow(int row) {this.row = row; }

	public void setBounds(Rect bounds) {
		this.bounds = bounds;
	}

	public Rect getBounds() {
		return bounds;
	}

	public Type getType() {
		return type;
	}
}
