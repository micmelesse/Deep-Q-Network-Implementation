import pygame

def show_screen(screen_data, screen):
    screen_data_rot = np.flipud(np.rot90(screen_data))
    screen.blit(pygame.pixelcopy.make_surface(screen_data_rot), (0, 0))
    pygame.display.flip()

# initialize pygame
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("DQN Agent Display")
pygame.display.flip()

show_screen(screen_data, screen) # call after every time screen_data gets updated
