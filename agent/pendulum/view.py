import numpy as np
import pygame
import copy

from agent.pendulum.config import SIMULATION
from agent.utils import wrap


class View(object):

    def __init__(self, hyperparams):
        config = copy.deepcopy(SIMULATION)
        config.update(hyperparams)
        self._hyperparams = config

        # screen parameter
        view_params = config['view']
        self._screen_size = view_params['screen']
        self._margin = view_params['margin']
        self._control_params = view_params['controller']
        self._C_ground = view_params['C_ground']
        self._C_info = view_params['C_info']
        self._C_margin = view_params['C_margin']
        self._C_body = view_params['C_body']
        self._C_x0 = view_params['C_x0']
        self._C_light_x0 = view_params['C_light_x0']
        self._C_tgt = view_params['C_tgt']

        pygame.init()
        self._clock = pygame.time.Clock()
        self._screen = None
        self._create_screen()
        self._fps = config['fps']

        # class properties
        self._l = config['model']['l']
        self._m = config['model']['m']
        self._max_torque = config['model']['max_torque']
        self._r_joint = view_params['r_joint']
        self._r_land_marker = view_params['r_land_marker']
        self._link_size = view_params['link_size']
        self._scale = view_params['scale_factor']
        self._cond_list = []  # List with all conditions

        # define new frame in the middle of the screen
        offset = self._screen_size[0] // 2
        self._frame = tuple([offset, self._screen_size[1] - offset])

    def register_condition(self, condition):
        self._cond_list.append(condition)

    def display_all(self, state, action, condition):
        """
        display full screen
        :param state: system state x = [theta, omega]
        :param action: applied torque u = [torque]
        :param condition: start condition
        :return: None
        """
        # clear screen
        self._draw_empty_screen()

        # draw margin for infos and action controller
        self._draw_margin()
        self._draw_info(state)

        # draw action controller
        self._draw_control_rail()
        action[action > self._max_torque] = self._max_torque
        action[action < -self._max_torque] = -self._max_torque
        self._draw_controller(action)

        '''draw individuals components'''
        # draw land marker of the inactive and active start conditions
        for cond in self._cond_list:
            theta = cond[0]
            # choose the right color light color for inactive conditions
            if cond[0] == condition[0] and cond[1] == condition[1]:
                color = self._C_x0
            else:
                color = self._C_light_x0
            px, py = self._scale * self._l * np.sin(theta), -self._scale * self._l * np.cos(theta)
            self._draw_point(px, py, color=color, r=self._r_land_marker)

        # draw target land marker
        px, py = 0.0, self._scale * self._l * 1.0
        self._draw_point(px, py, color=self._C_tgt, r=self._r_land_marker)

        # draw body
        self._draw_body(state)

        # limit the run speed of the game
        self._clock.tick_busy_loop(self._fps)

        # show window
        pygame.display.flip()

    def _draw_body(self, state):
        theta = state[0]
        px, py = self._scale * self._l * np.sin(theta), -self._scale * self._l * np.cos(theta)
        self._draw_link([0, 0], [px, py], color=self._C_body, width=self._link_size)
        self._draw_point(0, 0, color=self._C_body, r=self._r_joint)
        self._draw_point(px, py, color=self._C_body, r=self._r_joint)

    def _draw_empty_screen(self):
        self._screen.fill(self._C_ground)

    def _draw_margin(self):
        pygame.draw.rect(self._screen, self._C_margin,
                         (0, self._screen_size[1]-self._margin, self._screen_size[0], self._margin))

    def _draw_control_rail(self):
        start_pos = (
            self._screen_size[0]//2 - self._control_params['len_rail']//2,
            self._screen_size[1]-self._margin + self._control_params['off_rail']
        )
        end_pos = (
            self._screen_size[0]//2 + self._control_params['len_rail']//2,
            self._screen_size[1]-self._margin + self._control_params['off_rail']
        )
        pygame.draw.line(self._screen,
                         self._control_params['C_rail'],
                         start_pos, end_pos, self._control_params['w_rail']
                         )
        y_tick_start_pos = (self._screen_size[1] - self._margin
                            + self._control_params['off_rail'] - self._control_params['len_tick'] // 2)
        y_tick_end_pos = (self._screen_size[1] - self._margin
                          + self._control_params['off_rail'] + self._control_params['len_tick'] // 2)

        x_tick0_pos = start_pos[0]
        x_tick1_pos = self._screen_size[0]//2
        y_tick2_pos = end_pos[0]

        x0 = x_tick0_pos
        x1 = x_tick1_pos
        x2 = y_tick2_pos

        ys = y_tick_start_pos
        ye = y_tick_end_pos

        tick_list = [[(x0, ys), (x0, ye)], [(x1, ys), (x1, ye)], [(x2, ys), (x2, ye)]]
        for tick in tick_list:
            pygame.draw.line(self._screen,
                             self._control_params['C_rail'], tick[0], tick[1], self._control_params['w_rail'])

        self._screen.blit(pygame.font.SysFont('ubuntu', 16).render('0.0', False, self._control_params['C_rail']),
                          (
                              self._screen_size[0]//2 - 9,
                              self._screen_size[1]-self._margin + 1.2 * self._control_params['off_rail']
                          ))
        self._screen.blit(pygame.font.SysFont('ubuntu', 16).render('-u max', False, self._control_params['C_rail']),
                          (
                              start_pos[0] - 19,
                              self._screen_size[1] - self._margin + 1.2 * self._control_params['off_rail']
                          ))
        self._screen.blit(pygame.font.SysFont('ubuntu', 16).render('+u max', False, self._control_params['C_rail']),
                          (
                              end_pos[0] - 21,
                              self._screen_size[1] - self._margin + 1.2 * self._control_params['off_rail']
                          ))

    def _draw_controller(self, action):
        px = (self._screen_size[0]//2
              + int((action/self._hyperparams['model']['max_torque']) * (self._control_params['len_rail']//2)))
        py = self._screen_size[1] - self._margin + self._control_params['off_rail']
        if np.abs(action)[0] == self._max_torque:
            color = self._control_params['C_rail']
        else:
            color = self._C_info
        pygame.draw.circle(self._screen, color, (px, py), self._control_params['radius'])

    def _draw_info(self, state):
        theta, omega = state
        theta = wrap(theta + np.pi, m=-np.pi, M=np.pi)

        magnitude = abs(theta) * (180 / np.pi)
        info = f'Remaining Angle: {magnitude:>6.1f} °'

        self._screen.blit(pygame.font.SysFont('ubuntu', 18).render(info, False, self._C_info),
                          (12, self._screen_size[1] - (27 + 27))
                          )

        omega_deg = omega * (180 / np.pi)
        info = f'Angular Velocity: {omega_deg:>7.1f} °/sec.'
        self._screen.blit(pygame.font.SysFont('ubuntu', 18).render(info, False, self._C_info),
                          (12, self._screen_size[1] - 27)
                          )

    def _draw_point(self, px, py, color, r=1):
        sx, sy = self._transform_vector([px, py])
        pygame.draw.circle(self._screen, color, (sx, sy), r)

    def _draw_link(self, start_pos, end_pos, color, width):
        start_pos = self._transform_vector(start_pos)
        end_pos = self._transform_vector(end_pos)
        pygame.draw.aaline(self._screen, color, start_pos, end_pos)

    def _create_screen(self):
        self._screen = pygame.display.set_mode(self._screen_size)

    def _transform_vector(self, vec):
        sx = int(self._frame[0] + vec[0])
        sy = int(self._screen_size[1] - self._frame[1] - vec[1])
        return sx, sy