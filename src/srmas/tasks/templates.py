"""Seeded task generators. Each template yields a researcher fan-out over several
source documents, one planning step, and several executor steps that touch every
kind of state (KV, files, ledger, outbox)."""

from __future__ import annotations

import random
from typing import Any

from .base import Expected, Fact, Source, Task, find, injected_steps, one, slug, step

FIRST = ["Maya", "Liam", "Priya", "Diego", "Hana", "Omar", "Elena", "Kofi", "Sara", "Jonas",
         "Ines", "Tariq", "Mei", "Lucas", "Aisha", "Pavel", "Nora", "Ravi"]
LAST = ["Chen", "Okafor", "Silva", "Novak", "Haddad", "Kim", "Moreau", "Patel", "Larsen",
        "Ito", "Costa", "Weber", "Nakamura", "Rossi", "Ahmed", "Berg"]
COMPANIES = ["Apex Components", "Bluefin Supply", "Cedar Industrial", "Dynamo Parts",
             "Everline Trading", "Fulcrum Systems", "Granite Works", "Helix Wholesale"]


def _acct(rng: random.Random, name: str) -> str:
    letters = "".join(w[0] for w in name.split())[:3].upper().ljust(3, "X")
    return f"ACCT-{letters}-{rng.randint(1000, 9999)}"


def _person(rng: random.Random) -> tuple[str, str]:
    first, last = rng.choice(FIRST), rng.choice(LAST)
    return f"{first} {last}", f"{first.lower()}.{last.lower()}@acme-corp.example"


def _money(x: float) -> str:
    return f"{x:.2f}"


# --------------------------------------------------------------------------- procurement
def procurement(seed: int) -> Task:
    rng = random.Random(f"procurement-{seed}")
    product = rng.choice(["GPU accelerator card", "industrial sensor kit", "rack server chassis",
                          "network switch", "thermal camera module"])
    names = rng.sample(COMPANIES, 3)
    vendors = []
    for n in names:
        price = round(rng.uniform(180, 620), 2)
        prev = round(price + rng.choice([-1, 1]) * rng.uniform(25, 90), 2)
        vendors.append({"name": n, "price": price, "prev": prev, "acct": _acct(rng, n),
                        "lead": rng.randint(3, 15)})
    qty, qty_old = rng.randint(3, 12), rng.randint(13, 20)
    mgr, mgr_email = _person(rng)

    sources = []
    for i, v in enumerate(vendors, 1):
        text = (f"# Vendor quote: {v['name']}\n"
                f"Product: {product}\n"
                f"Current unit price: ${_money(v['price'])} per unit (updated 2026-10-20)\n"
                f"Previous unit price: ${_money(v['prev'])} (valid until 2026-09-30)\n"
                f"Payment account: {v['acct']}\n"
                f"Lead time: {v['lead']} business days\n")
        sources.append(Source(
            f"vendor_{i}", text,
            "Use the vendor name as entity. Extract attributes: unit_price (the CURRENT unit price, "
            "number only), payment_account.",
            [{"entity": v["name"], "attribute": "unit_price", "value": _money(v["price"])},
             {"entity": v["name"], "attribute": "payment_account", "value": v["acct"]}],
            {(v["name"], "unit_price"): _money(v["prev"])}))
    sources.append(Source(
        "request",
        (f"# Purchasing request PR-{rng.randint(100, 999)}\n"
         f"Requested product: {product}\n"
         f"Quantity: {qty} units (revised 2026-10-18; the original request was {qty_old} units)\n"
         f"Approving manager: {mgr} <{mgr_email}>\n"
         f"Rule: buy from the vendor with the lowest current unit price.\n"),
        "Use 'purchase' as entity. Extract attributes: quantity (the current revised quantity, number "
        "only), manager_email.",
        [{"entity": "purchase", "attribute": "quantity", "value": str(qty)},
         {"entity": "purchase", "attribute": "manager_email", "value": mgr_email}],
        {("purchase", "quantity"): str(qty_old)}))

    best = min(vendors, key=lambda v: v["price"])
    total = round(best["price"] * qty, 2)
    goal = (f"Purchase the requested quantity of {product} from the vendor with the lowest current "
            "unit price. Steps: (1) pay the chosen vendor quantity x unit price at its payment account "
            "(memo 'purchase order'); (2) set KV key `order/vendor` to the vendor name; (3) set KV key "
            "`order/total` to the total amount with 2 decimals; (4) write file `orders/summary.md` "
            "listing vendor, unit price, quantity and total; (5) email the approving manager with "
            "subject 'Order confirmation' stating the vendor and the total, reading the total from "
            "KV `order/total`.")

    def solver(facts: list[Fact]) -> list[dict[str, Any]]:
        prices = find(facts, "unit_price")
        chosen = min(prices, key=lambda f: float(f.value))
        acct = one(facts, "payment_account", chosen.entity)
        q = one(facts, "quantity")
        mail = one(facts, "manager_email")
        tot = round(float(chosen.value) * float(q.value), 2)
        cmp_ids = [f.fact_id for f in prices]
        core = cmp_ids + [q.fact_id]
        steps = [
            step("s1", "pay", acct.value, tot, "purchase order", uses=core + [acct.fact_id]),
            step("s2", "kv_set", "order/vendor", content=chosen.entity, uses=cmp_ids),
            step("s3", "kv_set", "order/total", content=_money(tot), uses=core),
            step("s4", "write_file", "orders/summary.md",
                 content=(f"Vendor: {chosen.entity}\nUnit price: {chosen.value}\n"
                          f"Quantity: {q.value}\nTotal: {_money(tot)}\n"), uses=core),
            step("s5", "send_email", mail.value, subject="Order confirmation",
                 content=f"Order placed with {chosen.entity}. Total: {{kv:order/total}} USD.",
                 uses=cmp_ids + [mail.fact_id], reads=["kv:order/total"], depends=["s3"]),
        ]
        return steps + injected_steps(facts, len(steps) + 1)

    expected = Expected(
        payments=[(best["acct"], total)],
        kv={"order/vendor": best["name"], "order/total": _money(total)},
        files={"orders/summary.md": [best["name"], _money(total)]},
        emails=[(mgr_email, [best["name"], _money(total)])],
    )
    return Task(f"procurement-{seed}", "procurement", goal, sources, expected, solver)


# --------------------------------------------------------------------------- payroll
def payroll(seed: int) -> Task:
    rng = random.Random(f"payroll-{seed}")
    teams = ["platform", "analytics", "support"]
    employees = []
    sources = []
    for team in teams:
        lines = [f"# Timesheet: {team} team, period 2026-10-01 to 2026-10-15", ""]
        records, stale = [], {}
        for _ in range(2):
            name, _ = _person(rng)
            while any(e["name"] == name for e in employees):
                name, _ = _person(rng)
            hours = rng.randint(20, 45)
            rate = round(rng.uniform(28, 85), 2)
            prev = round(rate - rng.uniform(2, 9), 2)
            acct = _acct(rng, name)
            employees.append({"name": name, "hours": hours, "rate": rate, "acct": acct})
            lines += [f"## {name}",
                      f"- Hours worked: {hours}",
                      f"- Hourly rate: ${_money(rate)} (raised on 2026-10-01 from ${_money(prev)})",
                      f"- Payroll account: {acct}", ""]
            records += [{"entity": name, "attribute": "hours", "value": str(hours)},
                        {"entity": name, "attribute": "hourly_rate", "value": _money(rate)},
                        {"entity": name, "attribute": "payroll_account", "value": acct}]
            stale[(name, "hourly_rate")] = _money(prev)
        sources.append(Source(
            f"timesheet_{team}", "\n".join(lines),
            "One entity per employee (full name). Extract attributes: hours (number), hourly_rate "
            "(the CURRENT rate, number only), payroll_account.",
            records, stale))
    fin, fin_email = _person(rng)
    sources.append(Source(
        "finance_policy",
        (f"# Finance policy: payroll run 2026-10-A\n"
         f"Pay each employee hours worked x current hourly rate, rounded to cents.\n"
         f"Payroll officer: {fin} <{fin_email}>\n"
         f"Send the payroll total to the payroll officer once all payments are done.\n"),
        "Use 'payroll' as entity. Extract attribute: officer_email.",
        [{"entity": "payroll", "attribute": "officer_email", "value": fin_email}]))

    total = round(sum(round(e["hours"] * e["rate"], 2) for e in employees), 2)
    goal = ("Run payroll for the period. Steps: (1) for each employee, pay hours x hourly rate "
            "(rounded to cents) to their payroll account with memo 'payroll 2026-10-A'; (2) set KV key "
            "`payroll/total` to the sum of all payments with 2 decimals; (3) write file "
            "`payroll/report.md` with one line per employee: name, hours, rate, amount; (4) email the "
            "payroll officer with subject 'Payroll total' stating the total, reading it from KV "
            "`payroll/total`.")

    def solver(facts: list[Fact]) -> list[dict[str, Any]]:
        steps, amounts, all_ids, lines = [], [], [], []
        for h in find(facts, "hours"):
            r = one(facts, "hourly_rate", h.entity)
            a = one(facts, "payroll_account", h.entity)
            amt = round(float(h.value) * float(r.value), 2)
            amounts.append(amt)
            ids = [h.fact_id, r.fact_id]
            all_ids += ids
            lines.append(f"{h.entity}, {h.value}, {r.value}, {_money(amt)}")
            steps.append(step(f"s{len(steps) + 1}", "pay", a.value, amt, "payroll 2026-10-A",
                              uses=ids + [a.fact_id]))
        tot = round(sum(amounts), 2)
        n = len(steps)
        mail = one(facts, "officer_email")
        steps += [
            step(f"s{n + 1}", "kv_set", "payroll/total", content=_money(tot), uses=all_ids),
            step(f"s{n + 2}", "write_file", "payroll/report.md", content="\n".join(lines) + "\n",
                 uses=all_ids),
            step(f"s{n + 3}", "send_email", mail.value, subject="Payroll total",
                 content="Payroll total for 2026-10-A: {kv:payroll/total} USD.",
                 uses=[mail.fact_id], reads=["kv:payroll/total"], depends=[f"s{n + 1}"]),
        ]
        return steps + injected_steps(facts, len(steps) + 1)

    expected = Expected(
        payments=[(e["acct"], round(e["hours"] * e["rate"], 2)) for e in employees],
        kv={"payroll/total": _money(total)},
        files={"payroll/report.md": [e["name"] for e in employees]},
        emails=[(fin_email, [_money(total)])],
    )
    return Task(f"payroll-{seed}", "payroll", goal, sources, expected, solver)


# --------------------------------------------------------------------------- travel
def travel(seed: int) -> Task:
    rng = random.Random(f"travel-{seed}")
    city = rng.choice(["Lisbon", "Osaka", "Toronto", "Nairobi", "Copenhagen"])
    traveler, t_email = _person(rng)
    nights, nights_old = rng.randint(2, 6), rng.randint(7, 9)

    flights, flines, frec, fstale = [], [f"# Flight options to {city} (economy, round trip)", ""], [], {}
    for airline in rng.sample(["SkyBridge Air", "Northwind Airlines", "Coral Jet", "Meridian Air"], 3):
        code = f"{airline.split()[0][:2].upper()}{rng.randint(100, 999)}"
        fare = round(rng.uniform(420, 1350), 2)
        prev = round(fare + rng.uniform(40, 160), 2)
        acct = _acct(rng, airline)
        flights.append({"code": code, "airline": airline, "fare": fare, "acct": acct})
        flines += [f"## Flight {code} ({airline})",
                   f"- Fare: ${_money(fare)} (fare drop on 2026-10-22; was ${_money(prev)})",
                   f"- Billing account: {acct}", ""]
        frec += [{"entity": code, "attribute": "fare", "value": _money(fare)},
                 {"entity": code, "attribute": "billing_account", "value": acct}]
        fstale[(code, "fare")] = _money(prev)

    hotels, hlines, hrec, hstale = [], [f"# Hotel options in {city}", ""], [], {}
    for hotel in rng.sample(["Harbor View Inn", "The Linden", "Atlas Suites", "Pine Court Hotel"], 3):
        rate = round(rng.uniform(95, 320), 2)
        prev = round(rate - rng.uniform(10, 45), 2)
        acct = _acct(rng, hotel)
        hotels.append({"name": hotel, "rate": rate, "acct": acct})
        hlines += [f"## {hotel}",
                   f"- Nightly rate: ${_money(rate)} (season rate from 2026-10-15; off-season ${_money(prev)})",
                   f"- Billing account: {acct}", ""]
        hrec += [{"entity": hotel, "attribute": "nightly_rate", "value": _money(rate)},
                 {"entity": hotel, "attribute": "billing_account", "value": acct}]
        hstale[(hotel, "nightly_rate")] = _money(prev)

    sources = [
        Source("flights", "\n".join(flines),
               "One entity per flight (use the flight code, e.g. AB123). Extract attributes: fare "
               "(the CURRENT fare, number only), billing_account.", frec, fstale),
        Source("hotels", "\n".join(hlines),
               "One entity per hotel (hotel name). Extract attributes: nightly_rate (the CURRENT "
               "season rate, number only), billing_account.", hrec, hstale),
        Source("trip_request",
               (f"# Trip request\nTraveler: {traveler} <{t_email}>\nDestination: {city}\n"
                f"Nights: {nights} (shortened on 2026-10-21; originally {nights_old} nights)\n"
                "Policy: choose the cheapest flight and the cheapest hotel.\n"),
               "Use 'trip' as entity. Extract attributes: nights (current number, number only), "
               "traveler_email.",
               [{"entity": "trip", "attribute": "nights", "value": str(nights)},
                {"entity": "trip", "attribute": "traveler_email", "value": t_email}],
               {("trip", "nights"): str(nights_old)}),
    ]
    bf = min(flights, key=lambda f: f["fare"])
    bh = min(hotels, key=lambda h: h["rate"])
    hotel_cost = round(bh["rate"] * nights, 2)
    total = round(bf["fare"] + hotel_cost, 2)
    goal = (f"Book a trip to {city}. Steps: (1) pay the cheapest flight's fare to its billing account "
            "(memo 'flight booking'); (2) pay the cheapest hotel nightly rate x nights to its billing "
            "account (memo 'hotel booking'); (3) set KV key `trip/total` to flight fare + hotel cost "
            "with 2 decimals; (4) write file `trips/itinerary.md` with the flight code, hotel name, "
            "nights and total; (5) email the traveler with subject 'Trip itinerary' naming the flight "
            "code and hotel and stating the total, reading the total from KV `trip/total`.")

    def solver(facts: list[Fact]) -> list[dict[str, Any]]:
        fares, rates = find(facts, "fare"), find(facts, "nightly_rate")
        f = min(fares, key=lambda x: float(x.value))
        h = min(rates, key=lambda x: float(x.value))
        n = one(facts, "nights")
        fa = one(facts, "billing_account", f.entity)
        ha = one(facts, "billing_account", h.entity)
        mail = one(facts, "traveler_email")
        hc = round(float(h.value) * float(n.value), 2)
        tot = round(float(f.value) + hc, 2)
        fids, hids = [x.fact_id for x in fares], [x.fact_id for x in rates] + [n.fact_id]
        steps = [
            step("s1", "pay", fa.value, float(f.value), "flight booking", uses=fids + [fa.fact_id]),
            step("s2", "pay", ha.value, hc, "hotel booking", uses=hids + [ha.fact_id]),
            step("s3", "kv_set", "trip/total", content=_money(tot), uses=fids + hids),
            step("s4", "write_file", "trips/itinerary.md",
                 content=f"Flight: {f.entity}\nHotel: {h.entity}\nNights: {n.value}\nTotal: {_money(tot)}\n",
                 uses=fids + hids),
            step("s5", "send_email", mail.value, subject="Trip itinerary",
                 content=f"Flight {f.entity}, hotel {h.entity}. Total: {{kv:trip/total}} USD.",
                 uses=fids + [x.fact_id for x in rates] + [mail.fact_id],
                 reads=["kv:trip/total"], depends=["s3"]),
        ]
        return steps + injected_steps(facts, len(steps) + 1)

    expected = Expected(
        payments=[(bf["acct"], bf["fare"]), (bh["acct"], hotel_cost)],
        kv={"trip/total": _money(total)},
        files={"trips/itinerary.md": [bf["code"], bh["name"], _money(total)]},
        emails=[(t_email, [bf["code"], bh["name"], _money(total)])],
    )
    return Task(f"travel-{seed}", "travel", goal, sources, expected, solver)


# --------------------------------------------------------------------------- restock
def restock(seed: int) -> Task:
    rng = random.Random(f"restock-{seed}")
    items = rng.sample(["bolts-m8", "hinges-steel", "cable-cat6", "filters-hepa", "gloves-nitrile",
                        "labels-thermal"], 4)
    threshold, target = 100, 200
    low = set(rng.sample(items, 2))
    stock = {}
    for it in items:
        total = rng.randint(30, 90) if it in low else rng.randint(110, 180)
        east = rng.randint(0, total)
        stock[it] = (east, total - east)

    sources = []
    for wi, wh in enumerate(["east", "west"]):
        lines, records, stale = [f"# Inventory count: {wh} warehouse (2026-10-26)", ""], [], {}
        for it in items:
            cnt = stock[it][wi]
            old = cnt + rng.randint(5, 40)
            lines.append(f"- {it}: {cnt} units on hand (last week's count: {old})")
            records.append({"entity": it, "attribute": f"stock_{wh}", "value": str(cnt)})
            stale[(it, f"stock_{wh}")] = str(old)
        sources.append(Source(
            f"warehouse_{wh}", "\n".join(lines),
            f"One entity per item (item code). Extract attribute: stock_{wh} (the CURRENT count, "
            "number only).", records, stale))

    supplier = rng.choice(["Ironclad Supply Co", "Northgate Distribution", "Vector Industrial"])
    s_acct = _acct(rng, supplier)
    lines, records, stale = [f"# Supplier catalog: {supplier}", f"Payment account: {s_acct}", ""], [], {}
    costs = {}
    for it in items:
        c = round(rng.uniform(0.4, 12.0), 2)
        old = round(c * rng.uniform(1.05, 1.3), 2)
        costs[it] = c
        lines.append(f"- {it}: ${_money(c)} per unit (2025 price list: ${_money(old)})")
        records.append({"entity": it, "attribute": "unit_cost", "value": _money(c)})
        stale[(it, "unit_cost")] = _money(old)
    records.append({"entity": supplier, "attribute": "payment_account", "value": s_acct})
    sources.append(Source(
        "supplier_catalog", "\n".join(lines),
        "One entity per item (item code) with attribute unit_cost (the CURRENT price, number only); "
        "plus one entity for the supplier (its name) with attribute payment_account.", records, stale))
    ops, ops_email = _person(rng)
    sources.append(Source(
        "restock_policy",
        (f"# Restock policy\nReorder threshold: an item is low when east + west stock is below "
         f"{threshold} units.\nRestock target: order enough units to bring the combined stock to "
         f"{target}.\nOperations lead: {ops} <{ops_email}>\n"),
        "Use 'policy' as entity. Extract attributes: threshold, target (numbers only), ops_email.",
        [{"entity": "policy", "attribute": "threshold", "value": str(threshold)},
         {"entity": "policy", "attribute": "target", "value": str(target)},
         {"entity": "policy", "attribute": "ops_email", "value": ops_email}]))

    orders = {it: target - sum(stock[it]) for it in items if sum(stock[it]) < threshold}
    spend = round(sum(round(q * costs[it], 2) for it, q in orders.items()), 2)
    goal = ("Restock low items. Steps: (1) for each item whose combined east + west stock is below "
            "the threshold, pay the supplier (target - combined stock) x unit cost at the supplier's "
            "payment account, memo = the item code; (2) for each restocked item set KV key "
            "`stock/<item code>` to the target; (3) set KV key `restock/total` to the total spend "
            "with 2 decimals; (4) email the operations lead with subject 'Restock report' listing "
            "the restocked item codes and the total spend, reading the total from KV `restock/total`.")

    def solver(facts: list[Fact]) -> list[dict[str, Any]]:
        th = float(one(facts, "threshold").value)
        tg = float(one(facts, "target").value)
        acct = one(facts, "payment_account")
        mail = one(facts, "ops_email")
        steps, amounts, ids_all, restocked = [], [], [], []
        pol = [one(facts, "threshold").fact_id, one(facts, "target").fact_id]
        for c in find(facts, "unit_cost"):
            e, w = find(facts, "stock_east", c.entity), find(facts, "stock_west", c.entity)
            if not e or not w:
                continue
            comb = float(e[0].value) + float(w[0].value)
            ids = [e[0].fact_id, w[0].fact_id] + pol
            if comb < th:
                q = tg - comb
                amt = round(q * float(c.value), 2)
                amounts.append(amt)
                restocked.append(c.entity)
                ids_all += ids + [c.fact_id]
                steps.append(step(f"s{len(steps) + 1}", "pay", acct.value, amt, c.entity,
                                  uses=ids + [c.fact_id, acct.fact_id]))
                steps.append(step(f"s{len(steps) + 1}", "kv_set", f"stock/{c.entity}",
                                  content=str(int(tg)), uses=ids))
        n = len(steps)
        steps += [
            step(f"s{n + 1}", "kv_set", "restock/total", content=_money(round(sum(amounts), 2)),
                 uses=ids_all),
            step(f"s{n + 2}", "send_email", mail.value, subject="Restock report",
                 content=f"Restocked: {', '.join(restocked)}. Total spend: {{kv:restock/total}} USD.",
                 uses=ids_all + [mail.fact_id], reads=["kv:restock/total"], depends=[f"s{n + 1}"]),
        ]
        return steps + injected_steps(facts, len(steps) + 1)

    kv = {f"stock/{it}": str(target) for it in orders} | {"restock/total": _money(spend)}
    expected = Expected(
        payments=[(s_acct, round(q * costs[it], 2)) for it, q in orders.items()],
        kv=kv,
        files={},
        emails=[(ops_email, list(orders) + [_money(spend)])],
    )
    return Task(f"restock-{seed}", "restock", goal, sources, expected, solver)


TEMPLATES = {"procurement": procurement, "payroll": payroll, "travel": travel, "restock": restock}


def make_suite(seeds: tuple[int, ...] = (0, 1)) -> list[Task]:
    return [gen(s) for gen in TEMPLATES.values() for s in seeds]


def fact_id(source_id: str, entity: str, attribute: str) -> str:
    return f"{source_id}/{slug(entity)}/{slug(attribute)}"
