//Name: Gilles Kepnang

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import java.util.Random;

public class GamePanel extends JPanel implements KeyListener 
{
    /**
     * Default serial version ID
     */
    private static final long serialVersionUID = 1L;
    
    private JLabel gameMap[][];
    private String gameArray[][];
    private JPanel menuPanel;
    private JPanel mapPanel;
    private JPanel scorePanel;
    private PacManCharacter pacMan;
    private JLabel scoreLabel;
    private JLabel menuLabel;
    private boolean pressed;
    
    public GamePanel()
    {  
        menuPanel = new JPanel();
        mapPanel = new JPanel();
        scorePanel = new JPanel();
        gameArray = new String[10][10];
        gameMap = new JLabel[10][10];
        
        /*
         * CREATE MENU PANEL
         */
        menuLabel = new JLabel();
        menuPanel.add(menuLabel);
        menuLabel.setText("Use arrow keys to move PacMan - game scores are tracked");
        
        /*
         * CREATE MAP PANEL
         */
        //Draw PacMan Grid or Map
        drawGrid();
        
        //Create PacMan
        pacMan = new PacManCharacter();
        pacMan.setDirection(">");
        
        //Create Game Array and place PacMan on Array
        createGameArray();
        placePacMan(0,0,pacMan.getDirection());
        updateGrid();        

        /*
         * CREATE SCORE PANEL
         */
        scoreLabel = new JLabel();
        scorePanel.add(scoreLabel);
        scoreLabel.setText("----------------------------------------------------");
        
        /*  ADD PANELS TO THE MAIN GAME PANEL    */

        //Set gridLayout for map panel
        mapPanel.setLayout(new GridLayout(10,10));
        mapPanel.setComponentOrientation(ComponentOrientation.LEFT_TO_RIGHT);

        //Border layouts for the main game panels
        BorderLayout b_layout = new BorderLayout();
        setLayout(b_layout);
        add(menuPanel, BorderLayout.NORTH);
        add(mapPanel, BorderLayout.CENTER);
        add(scorePanel, BorderLayout.SOUTH);
        
        //Add KeyListen for the Main Game Panel
        addKeyListener(this);
        setPressed(false);
    }

    public void addNotify() 
    {
        super.addNotify();
        requestFocus();
    }
    
    @Override
    public void keyPressed(KeyEvent e) // TODO Auto-generated method stub
    {
        /*  MAIN KEYCODES
         *   Left:   37
         *   Up:     38
         *   Right:  39
         *   Down;   40    
         */
        setPressed(true);
    }

    @Override
    public void keyReleased(KeyEvent e) // TODO Auto-generated method stub
    {
        if (pressed && ((pacMan.cookiesEaten + pacMan.spacesVisited < 99))) 
        {
            int keyCode = e.getKeyCode();
            setPressed(false);
            processReleasedKey(keyCode);
        }
        else if (pacMan.cookiesEaten + pacMan.spacesVisited == 99)
        {
            System.exit(0);
        }
    }

    @Override
    public void keyTyped(KeyEvent e) // TODO Auto-generated method stub
    {
        System.out.println("Current KeyEvent: " + e.getKeyCode());
    }
    
    public int[] generateCookies()
    {
        //Create cookie positions
        int cookieColumns[] = new int[3];
        int refInt = 0;
        
        //Generate three random numbers between 1 and 10
        Random rand = new Random(); 
        refInt = cookieColumns[0] = rand.nextInt(9);
        do 
        {
            refInt = rand.nextInt(10);
        }
        while (refInt == cookieColumns[0]);
        cookieColumns[1] =  refInt;
        do 
        {
            refInt = rand.nextInt(9);
        }
        while (refInt == cookieColumns[0] || refInt == cookieColumns[1]);
        cookieColumns[2] = refInt;
        
        return cookieColumns;
    }
    
    public void createGameArray()
    {
        int cookiePositions[];
        for (int i = 0; i < 10; i++)
        {
            cookiePositions = generateCookies();
            for (int j = 0; j < 10; j++)
            {
                if (j == cookiePositions[0] || 
                        j == cookiePositions[1] || 
                            j == cookiePositions[2])
                    gameArray[i][j] = "O";
                else gameArray[i][j] = ".";
            }     
        }
    }
    
    public void drawGrid()
    {   
      //Init game map for mapPanel
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++)
            {   
               gameMap[i][j] = new JLabel();
               mapPanel.add(gameMap[i][j]);
            }     
        }
    }
    
    public void updateGrid()
    {   
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++)
            {
               gameMap[i][j].setText(gameArray[i][j]);   
            }     
        }
    }
    
    public void updateScores()
    {
        String cookies = "Cookies: " + Integer.toString(pacMan.cookiesEaten) + "     ";
        String visited = "Spaces visited: " + Integer.toString(pacMan.spacesVisited);
        
        scoreLabel.setText(cookies + visited);

    }
    public void placePacMan(int x, int y, String dir)
    {
        gameArray[x][y] = dir;
    }

    public boolean isPressed() 
    {
        return pressed;
    }

    public void setPressed(boolean pressed) 
    {
        this.pressed = pressed;
    }
    
    public void processReleasedKey(int keyCode)
    {
        //Retrieve pacMan coordinates
        int x = pacMan.getPosX();
        int y = pacMan.getPosY();
        
        if (keyCode == 37 && y > 0)  //move left
        {
            moveLeftfrom(x,y);
        }
        else if (keyCode == 38 && x > 0) //move up
        {
            moveUpfrom(x,y);
        }
        else if (keyCode == 39 && y < 9) //move right
        {
            moveRightfrom(x,y);
        }
        else if (keyCode == 40 && x < 9) //move down
        {
            moveDownfrom(x,y);
        }
        else return;
    }
    
    public void moveLeftfrom(int x, int y)
    {
        //Determine new X and new Y
        int newX = x;
        int newY = y - 1;
        
        //Clear previous position
        gameArray[x][y] = " ";
        
        //Set pacMan's direction for "move left"
        pacMan.setPosX(newX);
        pacMan.setPosY(newY);
        pacMan.setDirection(">");
        
        //CHECK 3 CASES: 
        //1) Unvisited space (".")
        //2) Visited/Empty space (" ")
        //3) Cookie in the space ("O")
        if (gameArray[newX][newY] == ".")
            pacMan.spacesVisited++;
        else if (gameArray[newX][newY] == "O")
            pacMan.cookiesEaten++;
        
        placePacMan(newX,newY,pacMan.getDirection());
        updateGrid();
        updateScores();
    }
    
    public void moveUpfrom(int x, int y)
    {
        //Determine new X and new Y
        int newX = x - 1 ;
        int newY = y;
        
        //Clear previous position
        gameArray[x][y] = " ";
        
        //Set pacMan's direction for "move left"
        pacMan.setPosX(newX);
        pacMan.setPosY(newY);
        pacMan.setDirection("V");
        
        //CHECK 3 CASES: 
        //1) Unvisited space (".")
        //2) Visited/Empty space (" ")
        //3) Cookie in the space ("O")
        if (gameArray[newX][newY] == ".")
            pacMan.spacesVisited++;
        else if (gameArray[newX][newY] == "O")
            pacMan.cookiesEaten++;
        
        placePacMan(newX,newY,pacMan.getDirection());
        updateGrid();
        updateScores();
    }
    
    public void moveRightfrom(int x, int y)
    {
        //Determine new X and new Y
        int newX = x;
        int newY = y + 1;
        
        //Clear previous position
        gameArray[x][y] = " ";
        
        //Set pacMan's direction for "move left"
        pacMan.setPosX(newX);
        pacMan.setPosY(newY);
        pacMan.setDirection("<");
        
        //CHECK 3 CASES: 
        //1) Unvisited space (".")
        //2) Visited/Empty space (" ")
        //3) Cookie in the space ("O")
        if (gameArray[newX][newY] == ".")
            pacMan.spacesVisited++;
        else if (gameArray[newX][newY] == "O")
            pacMan.cookiesEaten++;
        
        placePacMan(newX,newY,pacMan.getDirection());
        updateGrid();
        updateScores();
    }
    
    public void moveDownfrom(int x, int y)
    {
        //Determine new X and new Y
        int newX = x + 1;
        int newY = y;
        
        //Clear previous position
        gameArray[x][y] = " ";
        
        //Set pacMan's direction for "move left"
        pacMan.setPosX(newX);
        pacMan.setPosY(newY);
        pacMan.setDirection("^");
        
        //CHECK 3 CASES: 
        //1) Unvisited space (".")
        //2) Visited/Empty space (" ")
        //3) Cookie in the space ("O")
        if (gameArray[newX][newY] == ".")
            pacMan.spacesVisited++;
        else if (gameArray[newX][newY] == "O")
            pacMan.cookiesEaten++;
        
        placePacMan(newX,newY,pacMan.getDirection());
        updateGrid();
        updateScores();
    }
}
