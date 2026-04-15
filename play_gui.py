"""
play_gui.py  —  Sequence Board Game GUI (pygame)
Run:  python play_gui.py
"""
import os, json, threading, queue, sys
import pygame

from shared.types import (BOARD_SIZE, CORNER_CHIP,
                           is_one_eyed_jack, is_two_eyed_jack, next_player)
from game.board import BOARD_LAYOUT
from game.game_loop import new_game, apply_move, handle_dead_cards
from game.moves import get_legal_moves

from game.agents.random_agent import RandomAgent
from agent.greedy_agent import GreedyAgent
from agent.combined_agent import CombinedAgent
from agent.search import minimax_search_with_eval
from agent.heuristic import evaluate

CELL  = 56
BSIZE = BOARD_SIZE * CELL          # 560
PW    = 220                        # panel width
PAD   = 10
WIN_W = PAD + BSIZE + PAD + PW + PAD   # 810
WIN_H = PAD + BSIZE + PAD              # 580

BG       = (55,  71,  79)
PANEL    = (69,  90, 100)
DARKER   = (38,  50,  56)
BTN      = (84, 110, 122)
BTN_HOV  = (96, 125, 139)
GREEN    = (67, 160,  71)
BLUE_BTN = (25, 118, 210)
RED_BTN  = (198,  40,  40)
GOLD     = (255, 215,   0)
ORANGE   = (255, 111,   0)

WHITE = (255,255,255);  LTGREY=(207,216,220)
P1C   = (100,181,246);  P2C=(239,154,154)
P1D   = ( 13, 71,161);  P2D=(183, 28, 28)
CRNR  = (144,164,174);  EMPT=(250,250,250)
VALID = (255,241,118)
TXT_DIM = (120,144,156);  TXT_BRIGHT=(255,255,255)

AGENTS = ["Human","Random","Greedy","Combined","RL-tuned","NN Agent"]
SUIT_SYM = {"spades":"S","hearts":"H","diamonds":"D","clubs":"C"}

def make_agent(name):
    if name == "Human":    return None
    if name == "Random":   return RandomAgent()
    if name == "Greedy":   return GreedyAgent()
    if name == "Combined": return CombinedAgent(n_samples=3, depth=2)
    if name == "RL-tuned":
        p = "training/learned_weights.json"
        if os.path.exists(p):
            w = json.load(open(p))["weights"]
            ev = lambda s,pl: evaluate(s,pl,weights=w)
            sr = lambda s,d,pl: minimax_search_with_eval(s,d,pl,ev)
            return CombinedAgent(n_samples=3, depth=2, search_fn=sr)
        return CombinedAgent(n_samples=3, depth=2)
    if name == "NN Agent":
        mp = "training/models/value_net_v2.pt"
        if os.path.exists(mp):
            try:
                from agent.nn_agent import NNAgent
                return NNAgent(model_path=mp, n_samples=2, depth=1)
            except Exception: pass
        return GreedyAgent()
    return None

def draw_rect_rounded(surf, color, rect, r=6, border=0, border_color=None):
    pygame.draw.rect(surf, color, rect, border_radius=r)
    if border:
        pygame.draw.rect(surf, border_color or WHITE, rect, border, border_radius=r)

def draw_text(surf, text, font, color, rect, align="center"):
    rendered = font.render(text, True, color)
    rr = rendered.get_rect()
    if align == "center":
        rr.center = (rect[0]+rect[2]//2, rect[1]+rect[3]//2)
    elif align == "left":
        rr.midleft = (rect[0]+6, rect[1]+rect[3]//2)
    elif align == "right":
        rr.midright = (rect[0]+rect[2]-6, rect[1]+rect[3]//2)
    surf.blit(rendered, rr)

def button(surf, text, font, rect, color=BTN, text_color=WHITE,
           hovered=False, radius=5):
    c = BTN_HOV if hovered else color
    draw_rect_rounded(surf, c, rect, radius)
    draw_text(surf, text, font, text_color, rect)
    return pygame.Rect(rect)

class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Sequence Board Game")
        self.clock  = pygame.time.Clock()

        # fonts
        self.f_big   = pygame.font.SysFont("Arial", 28, bold=True)
        self.f_med   = pygame.font.SysFont("Arial", 14, bold=True)
        self.f_small = pygame.font.SysFont("Arial", 12)
        self.f_tiny  = pygame.font.SysFont("Courier", 10)
        self.f_card  = pygame.font.SysFont("Courier", 11, bold=True)
        self.f_nav   = pygame.font.SysFont("Arial",   18, bold=True)

        self.screen_state = "MODE"   # "MODE" or "GAME"
        self.p1_idx = 0              # index into AGENTS
        self.p2_idx = 3             # "Combined" default

        self.states    = []
        self.moves     = []
        self.idx       = 0           # view index into history
        self.agents    = {}
        self.humans    = set()
        self.ai_vs_ai  = False
        self.over      = False
        self.auto      = True
        self.thinking  = False
        self.sel_card  = None
        self.valid_pos = set()
        self.legal_sel = []
        self.p1_name   = ""
        self.p2_name   = ""
        self._gen      = 0
        self.result_q  = queue.Queue()
        self.hover     = None        # tracks which button is hovered

        self.run()

    def run(self):
        while True:
            mouse_pos = pygame.mouse.get_pos()
            events = pygame.event.get()
            for ev in events:
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    self.on_click(ev.pos)

            # collect AI result if ready
            try:
                tag, data = self.result_q.get_nowait()
                if tag == "move":
                    gen, move = data
                    self.ai_done(move, gen)
                elif tag == "skip":
                    gen = data
                    self.ai_skip(gen)
            except queue.Empty:
                pass

            self.screen.fill(BG)
            if self.screen_state == "MODE":
                self.draw_mode(mouse_pos)
            else:
                self.draw_game(mouse_pos)

            pygame.display.flip()
            self.clock.tick(30)

    def draw_mode(self, mouse_pos):
        surf = self.screen
        cx = WIN_W // 2

        # title
        t = self.f_big.render("Sequence", True, WHITE)
        surf.blit(t, t.get_rect(centerx=cx, centery=100))
        t2 = self.f_small.render("Choose agents for each side", True, TXT_DIM)
        surf.blit(t2, t2.get_rect(centerx=cx, centery=140))

        # Player 1 selector
        self._draw_agent_row(surf, mouse_pos, "Player 1  (X — blue)",
                             self.p1_idx, 200, P1C)
        # Player 2 selector
        self._draw_agent_row(surf, mouse_pos, "Player 2  (O — red)",
                             self.p2_idx, 270, P2C)

        # Start button
        btn_r = pygame.Rect(cx-90, 360, 180, 46)
        hov   = btn_r.collidepoint(mouse_pos)
        button(surf, "> Start Game", self.f_med, btn_r,
               color=GREEN, hovered=hov)

    def _draw_agent_row(self, surf, mouse_pos, label, sel_idx, y, col):
        cx = WIN_W // 2
        # label
        t = self.f_small.render(label, True, col)
        surf.blit(t, t.get_rect(right=cx-10, centery=y+18))

        # prev arrow
        left_r  = pygame.Rect(cx+10, y, 32, 36)
        mid_r   = pygame.Rect(cx+46, y, 130, 36)
        right_r = pygame.Rect(cx+180, y, 32, 36)
        for r, txt in [(left_r,"<"),(right_r,">")]:
            hov = r.collidepoint(mouse_pos)
            draw_rect_rounded(surf, BTN_HOV if hov else BTN, r, 4)
            draw_text(surf, txt, self.f_nav, WHITE, r)
        draw_rect_rounded(surf, DARKER, mid_r, 4)
        draw_text(surf, AGENTS[sel_idx], self.f_small, WHITE, mid_r)

    def _mode_click(self, pos):
        cx = WIN_W // 2
        y1, y2 = 200, 270
        # p1 arrows
        if pygame.Rect(cx+10,  y1, 32, 36).collidepoint(pos):
            self.p1_idx = (self.p1_idx - 1) % len(AGENTS)
        elif pygame.Rect(cx+180, y1, 32, 36).collidepoint(pos):
            self.p1_idx = (self.p1_idx + 1) % len(AGENTS)
        # p2 arrows
        elif pygame.Rect(cx+10,  y2, 32, 36).collidepoint(pos):
            self.p2_idx = (self.p2_idx - 1) % len(AGENTS)
        elif pygame.Rect(cx+180, y2, 32, 36).collidepoint(pos):
            self.p2_idx = (self.p2_idx + 1) % len(AGENTS)
        # start
        elif pygame.Rect(WIN_W//2-90, 360, 180, 46).collidepoint(pos):
            self.start_game()

    def start_game(self):
        self._gen += 1
        self.p1_name = AGENTS[self.p1_idx]
        self.p2_name = AGENTS[self.p2_idx]
        self.agents  = {1: make_agent(self.p1_name), 2: make_agent(self.p2_name)}
        self.humans  = {pl for pl,a in self.agents.items() if a is None}
        self.ai_vs_ai = len(self.humans) == 0
        self.over     = False
        self.auto     = True
        self.thinking = False
        self.sel_card = None; self.valid_pos = set(); self.legal_sel = []

        st = new_game()
        handle_dead_cards(st, st.current_player)
        self.states = [st]; self.moves = []; self.idx = 0
        self.screen_state = "GAME"

        if self.live.current_player not in self.humans:
            self.trigger_ai(self._gen)

    def new_game(self):
        self._gen += 1
        self.over = False; self.thinking = False
        self.sel_card = None; self.valid_pos = set(); self.legal_sel = []
        self.screen_state = "MODE"

    def draw_game(self, mouse_pos):
        self.draw_board()
        self.draw_panel(mouse_pos)

    def draw_board(self):
        surf = self.screen
        st   = self.shown

        # last-played position
        last_pos = None
        if 0 < self.idx <= len(self.moves):
            try:
                raw = self.moves[self.idx-1].split("→")[1].split(")")[0].strip(" (")
                r,c = raw.split(","); last_pos = (int(r), int(c))
            except Exception: pass

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x = PAD + c * CELL
                y = PAD + r * CELL
                rect = pygame.Rect(x, y, CELL-1, CELL-1)
                pos  = (r, c)
                chip = int(st.chip_grid[r, c])
                seq  = pos in st.completed_sequences

                # cell fill
                if   pos in self.valid_pos:  fill = VALID
                elif chip == CORNER_CHIP:    fill = CRNR
                elif chip == 1: fill = P1D if seq else P1C
                elif chip == 2: fill = P2D if seq else P2C
                else:           fill = EMPT

                # border colour
                if   pos == last_pos: bdr = ORANGE; bw = 3
                elif seq:             bdr = GOLD;   bw = 2
                else:                 bdr = LTGREY; bw = 1

                pygame.draw.rect(surf, fill, rect)
                pygame.draw.rect(surf, bdr,  rect, bw)

                # card label
                card = BOARD_LAYOUT[r][c]
                if card is None:
                    t = self.f_small.render("*", True, BTN)
                    surf.blit(t, t.get_rect(center=rect.center))
                else:
                    tc = (200,0,0) if card.suit in ("hearts","diamonds") else (30,30,30)
                    if chip in (1, 2): tc = (200,225,255) if chip==1 else (255,210,210)
                    rank_t = self.f_tiny.render(card.rank, True, tc)
                    suit_t = self.f_tiny.render(SUIT_SYM[card.suit], True, tc)
                    surf.blit(rank_t, rank_t.get_rect(centerx=rect.centerx, top=rect.top+2))
                    surf.blit(suit_t, suit_t.get_rect(centerx=rect.centerx, bottom=rect.bottom-2))

                # chip circle
                if chip in (1, 2):
                    cr = CELL//2 - 8
                    cc_col = (P1D if seq else (30,100,220)) if chip==1 else (P2D if seq else (220,50,50))
                    pygame.draw.circle(surf, cc_col, rect.center, cr)
                    pygame.draw.circle(surf, WHITE, rect.center, cr, 1)
                    lbl = "X" if chip==1 else "O"
                    lt = self.f_card.render(lbl, True, WHITE)
                    surf.blit(lt, lt.get_rect(center=rect.center))

        # history banner
        if not self.at_live:
            bh = 22
            s = pygame.Surface((BSIZE, bh), pygame.SRCALPHA)
            s.fill((55,71,79,220))
            surf.blit(s, (PAD, PAD))
            msg = f"< HISTORY  move {self.idx} / {len(self.moves)} >"
            t = self.f_small.render(msg, True, GOLD)
            surf.blit(t, t.get_rect(centerx=PAD+BSIZE//2, centery=PAD+11))

    def draw_panel(self, mouse_pos):
        surf = self.screen
        px   = PAD + BSIZE + PAD          # panel x start
        py   = PAD                         # panel y start
        pw   = PW; ph = BSIZE

        # panel background
        pygame.draw.rect(surf, PANEL, (px, py, pw, ph), border_radius=6)

        cy = py + 10   # current y cursor

        # matchup
        t = self.f_med.render(f"{self.p1_name}  vs  {self.p2_name}", True, TXT_DIM)
        surf.blit(t, t.get_rect(centerx=px+pw//2, top=cy)); cy += 22

        # status
        status, scol = self._status_text()
        t = self.f_med.render(status, True, scol)
        surf.blit(t, t.get_rect(centerx=px+pw//2, top=cy)); cy += 20

        # sequence counts
        st = self.shown
        s1 = st.sequence_counts.get(1, 0)
        s2 = st.sequence_counts.get(2, 0)
        t = self.f_tiny.render(f"X({self.p1_name[:8]}): {s1}/2   O({self.p2_name[:8]}): {s2}/2",
                               True, TXT_DIM)
        surf.blit(t, t.get_rect(centerx=px+pw//2, top=cy)); cy += 18

        # separator
        pygame.draw.line(surf, BTN, (px+8, cy), (px+pw-8, cy)); cy += 8

        # hand (human turn only)
        pl = self.live.current_player
        if self.at_live and pl in self.humans and not self.over:
            t = self.f_small.render(f"Player {pl}'s Hand", True,
                                    P1C if pl==1 else P2C)
            surf.blit(t, (px+8, cy)); cy += 18
            for card in self.live.hands.get(pl, []):
                sel  = card == self.sel_card
                crect = pygame.Rect(px+6, cy, pw-12, 22)
                col   = GOLD if sel else BTN
                draw_rect_rounded(surf, col, crect, 3)
                lbl = self._card_label(card)
                draw_text(surf, lbl, self.f_card,
                          (30,30,30) if sel else WHITE, crect, "left")
                cy += 24
            # hint
            if self.sel_card:
                hint = self._hint()
                if hint:
                    t = self.f_tiny.render(hint, True, GOLD)
                    surf.blit(t, t.get_rect(centerx=px+pw//2, top=cy)); cy += 16
            pygame.draw.line(surf, BTN, (px+8, cy), (px+pw-8, cy)); cy += 8

        # move history nav — pinned near bottom
        nav_y = py + ph - (46 + 28 + 28 + 8 + (36 if self.ai_vs_ai else 0))
        pygame.draw.line(surf, BTN, (px+8, nav_y-4), (px+pw-8, nav_y-4))

        t = self.f_tiny.render("Move History", True, TXT_DIM)
        surf.blit(t, t.get_rect(centerx=px+pw//2, top=nav_y)); nav_y += 16

        n = len(self.moves)
        t = self.f_tiny.render(f"Move {self.idx} / {n}", True, TXT_DIM)
        surf.blit(t, t.get_rect(centerx=px+pw//2, top=nav_y)); nav_y += 16

        bw = 44; gap = 2; nb = 4
        total = nb*bw + (nb-1)*gap
        nx0 = px + (pw - total)//2
        for i, (sym, tag) in enumerate([("|<","first"),("<","prev"),(">","next"),(">|","last")]):
            br = pygame.Rect(nx0 + i*(bw+gap), nav_y, bw, 26)
            can = self._nav_can(tag)
            col = BTN if can else DARKER
            hov = br.collidepoint(mouse_pos) and can
            button(surf, sym, self.f_nav, br, color=col, hovered=hov)
        nav_y += 30

        # pause button
        if self.ai_vs_ai:
            pr = pygame.Rect(px+8, nav_y, pw-16, 28)
            pp_txt = "|| Pause" if self.auto else "> Resume"
            pp_col = BLUE_BTN if self.auto else GREEN
            hov = pr.collidepoint(mouse_pos)
            button(surf, pp_txt, self.f_small, pr, color=pp_col, hovered=hov)
            nav_y += 32

        # last move description
        if 0 < self.idx <= len(self.moves):
            desc = self.moves[self.idx-1]
            t = self.f_tiny.render(desc[:34], True, TXT_DIM)
            surf.blit(t, t.get_rect(centerx=px+pw//2, top=nav_y)); nav_y += 14
            if len(desc) > 34:
                t2 = self.f_tiny.render(desc[34:68], True, TXT_DIM)
                surf.blit(t2, t2.get_rect(centerx=px+pw//2, top=nav_y)); nav_y += 14

        # new game button
        ng_r = pygame.Rect(px+8, py+ph-34, pw-16, 28)
        hov  = ng_r.collidepoint(mouse_pos)
        button(surf, "New Game", self.f_small, ng_r, color=BTN, hovered=hov)

    def on_click(self, pos):
        if self.screen_state == "MODE":
            self._mode_click(pos)
            return

        # new game button
        px = PAD + BSIZE + PAD
        ng_r = pygame.Rect(px+8, PAD+BSIZE-34, PW-16, 28)
        if ng_r.collidepoint(pos):
            self.new_game(); return

        # pause
        if self.ai_vs_ai:
            nav_y = PAD + BSIZE - (46 + 28 + 28 + 8 + 36)
            nav_y += 48   # offset to pause button position
            pr = pygame.Rect(px+8, nav_y, PW-16, 28)
            if pr.collidepoint(pos):
                self.toggle_pause(); return

        # nav buttons
        nav_y0 = PAD + BSIZE - (46 + 28 + 28 + 8 + (36 if self.ai_vs_ai else 0))
        nav_y0 += 32   # offset to nav buttons
        bw=44; gap=2; nb=4; total=nb*bw+(nb-1)*gap
        nx0 = px + (PW - total)//2
        for i, tag in enumerate(["first","prev","next","last"]):
            br = pygame.Rect(nx0 + i*(bw+gap), nav_y0, bw, 26)
            if br.collidepoint(pos) and self._nav_can(tag):
                self._nav_action(tag); return

        # board click
        bx = pos[0] - PAD; by = pos[1] - PAD
        if 0 <= bx < BSIZE and 0 <= by < BSIZE:
            c, r = bx // CELL, by // CELL
            self._board_click(r, c); return

        # hand cards
        pl = self.live.current_player
        if self.at_live and pl in self.humans and not self.over:
            card_y_start = self._hand_y_start()
            for i, card in enumerate(self.live.hands.get(pl, [])):
                crect = pygame.Rect(px+6, card_y_start + i*24, PW-12, 22)
                if crect.collidepoint(pos):
                    self._pick_card(card); return

    def _hand_y_start(self):
        # same y calculation as draw_panel for hand cards
        py = PAD
        cy = py + 10 + 22 + 20 + 18 + 8 + 18   # after header items + sep + "Hand" label
        return cy

    def _board_click(self, r, c):
        if (not self.states or not self.at_live or self.thinking
                or self.over or self.live.current_player not in self.humans):
            return
        if self.sel_card is None: return
        matches = [m for m in self.legal_sel if m.position == (r,c)]
        if matches: self._do_move(matches[0])

    def _pick_card(self, card):
        if self.thinking or not self.at_live: return
        if self.sel_card == card:
            self.sel_card = None; self.valid_pos = set(); self.legal_sel = []
        else:
            self.sel_card  = card
            legal = get_legal_moves(self.live, self.live.current_player)
            self.legal_sel = [m for m in legal if m.card == card]
            self.valid_pos = {m.position for m in self.legal_sel}

    def _do_move(self, move):
        self.sel_card = None; self.valid_pos = set(); self.legal_sel = []
        winner = self._apply(move)
        if winner:
            self.over = True
            self._show_winner(winner)
        elif self.live.current_player not in self.humans:
            self.trigger_ai(self._gen)

    def _apply(self, move) -> int:
        prev = self.live; pl = prev.current_player
        ns   = prev.copy(); winner = apply_move(ns, move)
        if not winner: handle_dead_cards(ns, ns.current_player)
        s = SUIT_SYM[move.card.suit]; r,c = move.position
        nm = self.p1_name if pl==1 else self.p2_name
        self.moves.append(f"P{pl}({nm}): {move.card.rank}{s}->({r},{c})[{move.move_type}]")
        self.states.append(ns)
        self.idx = len(self.states)-1
        return winner

    def trigger_ai(self, gen):
        if gen != self._gen or self.over or not self.at_live: return
        pl = self.live.current_player
        ag = self.agents.get(pl)
        if ag is None: return
        self.thinking = True
        snap = self.live
        def run():
            if not get_legal_moves(snap, pl):
                self.result_q.put(("skip", gen)); return
            m = ag.choose_move(snap)
            self.result_q.put(("move", (gen, m)))
        threading.Thread(target=run, daemon=True).start()

    def ai_done(self, move, gen):
        if gen != self._gen or self.over: return
        self.thinking = False
        was_live = self.at_live; saved = self.idx
        winner = self._apply(move)
        if not was_live: self.idx = saved
        if winner:
            self.over = True
            self._show_winner(winner); return
        nxt = self.live.current_player
        g   = self._gen
        if nxt not in self.humans and self.auto and self.at_live:
            pygame.time.set_timer(pygame.USEREVENT+1, 450 if self.ai_vs_ai else 300,
                                  loops=1)
            # store gen for timer
            self._pending_ai_gen = g

    def ai_skip(self, gen):
        if gen != self._gen: return
        self.thinking = False
        ns = self.live.copy()
        ns.current_player = next_player(ns.current_player)
        self.states.append(ns); self.moves.append("(skipped)"); self.idx = len(self.states)-1
        if self.auto: self.trigger_ai(self._gen)

    def _nav_can(self, tag):
        if tag in ("first","prev"): return self.idx > 0
        if tag in ("next","last"):  return self.idx < len(self.states)-1
        return False

    def _nav_action(self, tag):
        if tag == "first": self.idx = 0
        elif tag == "prev": self.idx = max(0, self.idx-1)
        elif tag == "last": self.idx = len(self.states)-1
        elif tag == "next":
            if self.idx < len(self.states)-1:
                self.idx += 1
            elif self.ai_vs_ai and not self.thinking and not self.over and not self.auto:
                self.trigger_ai(self._gen)

    def toggle_pause(self):
        self.auto = not self.auto
        if self.auto and self.at_live and not self.thinking and not self.over:
            self.trigger_ai(self._gen)

    @property
    def live(self):    return self.states[-1]
    @property
    def shown(self):   return self.states[self.idx]
    @property
    def at_live(self): return self.idx == len(self.states)-1

    def _status_text(self):
        if not self.at_live:
            return "< viewing history >", GOLD
        if self.over:
            return "Game over", TXT_DIM
        pl = self.live.current_player
        if self.thinking:
            n = self.p1_name if pl==1 else self.p2_name
            return f"{n} thinking...", GOLD
        if pl in self.humans:
            sym = "X" if pl==1 else "O"
            return f"Player {pl}'s turn ({sym})", P1C if pl==1 else P2C
        n = self.p1_name if pl==1 else self.p2_name
        return f"{n} thinking...", GOLD

    def _card_label(self, card):
        s = SUIT_SYM[card.suit]
        if is_two_eyed_jack(card): return f"J{s} wild"
        if is_one_eyed_jack(card): return f"J{s} rmv"
        return f"{card.rank}{s}"

    def _hint(self):
        if not self.sel_card: return ""
        if not self.valid_pos: return "No valid positions"
        if is_two_eyed_jack(self.sel_card): return "Click any empty cell"
        if is_one_eyed_jack(self.sel_card): return "Click opponent chip"
        return "Click highlighted cell"

    def _show_winner(self, winner):
        nm = self.p1_name if winner==1 else self.p2_name
        # draw a simple overlay
        s = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        s.fill((0,0,0,160))
        self.screen.blit(s, (0,0))
        msg1 = f"{nm} wins!" if winner not in self.humans else "You win!"
        msg2 = f"Player {winner} completed 2 sequences"
        t1 = self.f_big.render(msg1, True, GOLD)
        t2 = self.f_small.render(msg2, True, WHITE)
        t3 = self.f_small.render("Click  New Game  to play again", True, TXT_DIM)
        cy = WIN_H//2 - 50
        for t in [t1, t2, t3]:
            self.screen.blit(t, t.get_rect(centerx=WIN_W//2, centery=cy)); cy += 36
        pygame.display.flip()


if __name__ == "__main__":
    # handle AI timer events
    orig_run = App.run
    def patched_run(self):
        self._pending_ai_gen = -1
        while True:
            mouse_pos = pygame.mouse.get_pos()
            events = pygame.event.get()
            for ev in events:
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    self.on_click(ev.pos)
                if ev.type == pygame.USEREVENT+1:
                    if self._pending_ai_gen == self._gen:
                        self.trigger_ai(self._gen)

            try:
                tag, data = self.result_q.get_nowait()
                if tag == "move":
                    gen, move = data; self.ai_done(move, gen)
                elif tag == "skip":
                    self.ai_skip(data)
            except queue.Empty:
                pass

            self.screen.fill(BG)
            if self.screen_state == "MODE":
                self.draw_mode(mouse_pos)
            else:
                self.draw_game(mouse_pos)
            pygame.display.flip()
            self.clock.tick(30)

    App.run = patched_run
    App()
