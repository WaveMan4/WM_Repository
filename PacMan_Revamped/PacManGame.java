/*--------------------------------------------------
 *                  PACMAN GAME                    |
 * -------------------------------------------------                 
 * AUTHORED  BY: Gilles Kepnang
 * IMPLEMENTED: May 2017
 * 
 * SUMMARY: This game was designed with Java Swing Framework.
 * ***
 * ***
 * ***
 * ***
 */

import javax.swing.JFrame;
import javax.swing.JPanel;

public class PacManGame extends JFrame
{
    /**
     * Default serial version ID
     */
    private static final long serialVersionUID = 1L;

    /**
     * Default Constructor
     */
    public PacManGame()
    {
        setTitle("PACMAN - ReVAMPED");
        setSize(500, 500);
        setDefaultCloseOperation( JFrame.EXIT_ON_CLOSE );
        JPanel GamePanel = new GamePanel();
        
        this.add( GamePanel );
    }
    
    
    
    ////////////////////////////////////////////////////
    //                MAIN FUNCTION                   //
    ////////////////////////////////////////////////////
    /**
     * Main Function for GUI program 
     */
    public static void main (String [] args)
    {        
        JFrame mainFrame = new PacManGame();
        mainFrame.setVisible(true);
        return; 
    }
}

