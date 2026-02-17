from clingo.control import Control
from clingo.symbol import Number

ENCODING = """ 
hyp(@cfg_idxs()).
env(@env_idxs()).
regret(C,E,@regret(C,E)) :- hyp(C), env(E).

:- hyp(H), env(E), not regret(H,E,_). % is data complete?

{ select(H) : hyp(H) } = 2.
{ assignment(H, E) : select(H) } = 1 :- env(E).
:- assignment(H1, E), assignment(H2, E), H1 != H2.

#minimize { V,E,H : assignment(H,E), regret(H,E,V) }.
#show select/1.
"""


class Context:
    def cfg_idxs(self):
        return [Number(i) for i in range(5)]

    def env_idxs(self):
        return [Number(i) for i in range(20)]

    def regret(self, cfg_idx, env_idx):
        r = 29 * cfg_idx.number + 3 * env_idx.number
        return Number(int(r % 100))


ctl = Control(["0"])
ctl.add("base", [], ENCODING)
ctl.ground([("base", [])], context=Context())

TIMEOUT_SECONDS = 5.0

with ctl.solve(async_=True) as handle:
    finished = handle.wait(TIMEOUT_SECONDS)
    if not finished:
        print("Timeout reached — cancelling search.")
        handle.cancel()
    result = handle.get()

last_model = handle.last()
if last_model is not None:
    maybe_not = " not" if not last_model.optimality_proven else ""
    print(
        f"Found {last_model} with cost {last_model.cost[0]} and "
        f"optimality{maybe_not} proven."
    )
else:
    print("No model was produced (unsat or interrupted).")
