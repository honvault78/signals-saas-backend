"""Decision brief renderer. Displays existing engine outputs; never decides trades.

Keep report_template.html beside this module. No external frontend dependencies.
The original generate_html_report arguments remain supported.
"""
from __future__ import annotations
from datetime import datetime
from html import escape
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Optional


def _num(value):
    if isinstance(value, bool): return None
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError): return None


def _clean(value):
    if isinstance(value, dict): return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_clean(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)): return value
    if hasattr(value, 'item'): return _clean(value.item())
    return _num(value)


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip, self.links = [], 0, []
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'): self.skip += 1
        if self.skip: return
        if tag in ('p', 'div', 'h1', 'h2', 'h3', 'h4', 'li', 'br', 'tr'): self.parts.append('\n')
        if tag == 'a':
            href = dict(attrs).get('href', '')
            self.links.append(href if href.startswith(('https://', 'http://')) else '')
    def handle_endtag(self, tag):
        if tag in ('script', 'style'): self.skip = max(0, self.skip - 1)
        elif not self.skip and tag == 'a' and self.links:
            href = self.links.pop()
            if href: self.parts.append(' (' + href + ')')
        elif not self.skip and tag in ('p', 'div', 'li', 'tr'): self.parts.append('\n')
    def handle_data(self, data):
        if not self.skip: self.parts.append(data)


def _plain(value):
    p = _Text(); p.feed(str(value or ''))
    return re.sub(r'\n[ \t]*\n+', '\n\n', ''.join(p.parts)).strip()


def _date(value):
    if value is None: return None
    try:
        day = value.strftime('%Y-%m-%d') if hasattr(value, 'strftime') else str(value)[:10]
        datetime.strptime(day, '%Y-%m-%d')
        return day
    except (ValueError, TypeError): return None


def _chart_data(cumulative, z_score, regimes, window, has_dates):
    """Exact engine z-scores, without warm-up fill zeros or invented snapshots."""
    if not has_dates or cumulative is None or z_score is None: return [], []
    try:
        window = int(window)
        if window < 2 or not hasattr(z_score, 'items'): return [], []
        std = cumulative.rolling(window=window).std()
        points = {}
        for index, value in z_score.items():
            day, n, scale = _date(index), _num(value), _num(std.get(index))
            if day and n is not None and scale is not None and scale > 0:
                points[day] = {'date': day, 'value': n}
        history = [points[day] for day in sorted(points)]
        events = []
        if history and regimes is not None and 'regime' in regimes:
            runs, previous = [], None
            for index, row in regimes.iterrows():
                day, current = _date(index), str(row['regime'])
                if current.lower() in ('nan', 'none', 'unknown') or not day: previous = None; continue
                if runs and current == previous: runs[-1]['length'] += 1
                else: runs.append({'date': day, 'regime': current, 'length': 1})
                previous = current
            # Only regimes that persisted for at least five observations are worth annotating.
            stable = [r for r in runs if r['length'] >= 5]
            for prev, cur in zip(stable, stable[1:]):
                if history[0]['date'] <= cur['date'] <= history[-1]['date']:
                    events.append({'date': cur['date'], 'title': 'Market backdrop changed', 'topic': 'timing',
                                   'summary': f"Rolling regime moved from {prev['regime'].replace('_',' ')} to {cur['regime'].replace('_',' ')} and held for {cur['length']} observations. A regime observation, not a decision."})
        return history, events
    except (AttributeError, TypeError, ValueError, KeyError): return [], []


def _clean_name(value, ticker):
    """Trim provider padding ("Siemens Energy AG             N") and shouting caps."""
    name = re.sub(r'\s+', ' ', str(value or '')).strip()
    name = re.sub(r'\s+N$', '', name)
    if not name: return str(ticker)
    if name.isupper() and len(name) > 4: name = name.title()
    return name


_STANCE_RE = re.compile(r'FUNDAMENTAL\s+STANCE\s*[:\-—]\s*\**\s*(SUPPORTS|NEUTRAL|IN\s+TENSION)\**(?:.*?CONVICTION\s*[:\-—]\s*\**\s*(LOW|MEDIUM|MED|HIGH))?', re.I | re.S)
_VERDICT_RE = re.compile(r'Fundamentals\s+favou?r\s+(?:the\s+)?\**\s*(LONG|SHORT|NEITHER)', re.I)
_CONV_RE = re.compile(r'CONVICTION\s*[:\-—]\s*\**\s*(LOW|MEDIUM|MED|HIGH)', re.I)
_DATE_RE = re.compile(r'(20\d\d-\d\d-\d\d)\s*[:\-—]+\s*([^\n]{6,140})')


def _stance(memo_text, fund_text, is_equity_pair):
    """Fundamental stance relative to the position as held: for / neutral / against, plus conviction."""
    if not is_equity_pair: return {'state': 'n/a', 'conviction': None, 'source': None}
    conviction = None
    for text in (memo_text, fund_text):
        m = _CONV_RE.search(text or '')
        if m: conviction = {'MED': 'MEDIUM'}.get(m.group(1).upper(), m.group(1).upper()); break
    m = _STANCE_RE.search(memo_text or '')
    if m:
        word = re.sub(r'\s+', ' ', m.group(1).upper())
        state = {'SUPPORTS': 'for', 'NEUTRAL': 'neutral', 'IN TENSION': 'against'}[word]
        if m.group(2): conviction = {'MED': 'MEDIUM'}.get(m.group(2).upper(), m.group(2).upper())
        return {'state': state, 'conviction': conviction, 'source': 'stance'}
    for text, source in ((fund_text, 'verdict'), (memo_text, 'memo')):
        m = _VERDICT_RE.search(text or '')
        if m:
            state = {'LONG': 'for', 'SHORT': 'against', 'NEITHER': 'neutral'}[m.group(1).upper()]
            return {'state': state, 'conviction': conviction, 'source': source}
    return {'state': 'unknown', 'conviction': conviction, 'source': None}


def _next_catalyst(*texts, after=None):
    """Earliest dated catalyst found in the research, strictly after the as-of date."""
    best = None
    for text in texts:
        for day, title in _DATE_RE.findall(text or ''):
            if after and day <= after: continue
            title = re.sub(r'[*_#]+', '', title).strip(' .—-:')
            if best is None or day < best[0]: best = (day, title)
    return {'title': best[1], 'date': best[0], 'dateVerified': True} if best else None


VERB_ORDER = ['Build', 'Hold', 'Reduce', 'Exit']


def _decide(engine_action, health_state, stats, fundamentals, risk, adf_p, z_last):
    """Combine the engine decision with the fundamental stance and realised risk into one verb.

    The engine already encodes validity, failure modes, regime and P&L. Fundamentals and risk can
    only pull the verb down; a strong statistical setup can push Hold up to Build. Every step is
    explained in `rule` so the reader can check it.
    """
    base = {'ENTER': 'Build', 'HOLD': 'Hold', 'REDUCE': 'Reduce', 'WAIT': 'Reduce', 'EXIT': 'Exit', 'REVERSE': 'Exit'}.get(engine_action, 'Hold')
    steps = [f'Engine view: {base}.']
    fund_against = fundamentals['state'] == 'against'
    stats_against = stats['state'] == 'against'
    stats_strong = stats['state'] == 'for' and z_last is not None and z_last <= -2
    stretched = risk['state'] == 'stretched'
    verb = base
    if health_state in ('BROKEN', 'INVALID'):
        verb = 'Exit'; steps.append('The relationship is broken, so the only answer is Exit.')
    elif base == 'Build' and (fund_against or stretched):
        verb = 'Hold'; steps.append('Fundamentals or realised risk argue against adding, so Build becomes Hold.')
    elif base == 'Hold':
        if stats_against and fund_against:
            verb = 'Exit'; steps.append('Neither the statistics nor the fundamentals support the position, so Hold becomes Exit.')
        elif fund_against or stretched or stats_against:
            pulls = [n for n, f in (('fundamentals', fund_against), ('realised risk', stretched), ('statistics', stats_against)) if f]
            verb = 'Reduce'; steps.append(f'{" and ".join(pulls).capitalize()} pull against the position, so Hold becomes Reduce.')
        elif stats_strong and not fund_against and not stretched:
            verb = 'Build'; steps.append('The spread is beyond 2σ against the position with nothing arguing against it, so Hold becomes Build.')
        else: steps.append('Nothing pulls the view up or down.')
    elif base == 'Reduce' and stats_against and fund_against:
        verb = 'Exit'; steps.append('Neither the statistics nor the fundamentals support the position, so Reduce becomes Exit.')
    else: steps.append('Nothing changes the engine view.')
    idea = 'Build' if (verb in ('Build',) or (verb == 'Hold' and stats_strong and health_state == 'VALID' and not fund_against)) else 'Hold off'
    return verb, idea, ' '.join(steps)


# Taxonomy from this project's bavella_adapter.py, not the illustrative demo.
FM = {
    'FM1': ('Volatility regime shift', 'Price swings have changed.'),
    'FM2': ('Parameter drift', 'The relationship’s parameters are drifting.'),
    'FM3': ('Seasonality mismatch', 'The usual timing is less reliable.'),
    'FM4': ('Structural break', 'A sudden structural change is reported.'),
    'FM5': ('Outlier contamination', 'Extreme observations are influencing the estimates.'),
    'FM6': ('Extreme positioning', 'The portfolio path is unusually far from its reference.'),
    'FM7': ('Dependency break', 'The assets’ dependencies have changed.'),
}


def generate_html_report(
    enhanced_stats: Dict[str, Any], regime_summary: Dict[str, Any],
    memo_text: Optional[str] = None, long_positions: Optional[Dict[str, float]] = None,
    short_positions: Optional[Dict[str, float]] = None, portfolio_name: str = 'Long/Short Equity Portfolio',
    regime_chart_base64: Optional[str] = None, performance_chart_base64: Optional[str] = None,
    distribution_chart_base64: Optional[str] = None, validity_data: Optional[Dict[str, Any]] = None,
    analysis_period_days: int = 180, claude_fs_html: Optional[str] = None,
    deterministic_decision: Optional[Dict[str, Any]] = None,
    chart_z_score=None, chart_cumulative=None, chart_regime_history=None, chart_window: int = 60,
    as_of_date=None, position_gross_exposure=None, fundamental_data=None,
    is_equity_pair: bool = False, analysis_has_dates: bool = True, risk_horizon_days: Optional[int] = 1,
) -> str:
    stats, regime, raw = enhanced_stats or {}, regime_summary or {}, validity_data or {}
    v, details, attr, engine = raw.get('validity', raw) or {}, raw.get('details', {}) or {}, raw.get('attribution', {}) or {}, raw.get('engine', {}) or {}
    state, score = str(v.get('state') or 'UNKNOWN').upper(), _num(v.get('score'))
    dd = deterministic_decision or {}
    action = str(dd.get('decision') or 'NOT ASSESSED').upper()
    rationale = str(dd.get('rationale') or 'A deterministic position assessment was not supplied for this analysis.')
    health = {'VALID':'Relationship intact', 'DEGRADED':'Relationship under strain', 'BROKEN':'Relationship broken', 'INVALID':'Relationship broken'}.get(state, 'Relationship not assessed')
    longs, shorts, funds = long_positions or {}, short_positions or {}, fundamental_data or {}
    def asset(positions, side):
        if len(positions) == 1:
            ticker = next(iter(positions))
            return {'name': _clean_name((funds.get(ticker) or {}).get('name'), ticker), 'ticker': ticker}
        return {'name': f'{side} basket ({len(positions)})' if positions else f'No {side.lower()} leg', 'ticker': ', '.join(positions)}
    series, events = _chart_data(chart_cumulative, chart_z_score, chart_regime_history, chart_window, analysis_has_dates)
    asof = _date(as_of_date)
    if analysis_has_dates and chart_cumulative is not None and len(chart_cumulative): asof = _date(chart_cumulative.index[-1]) or asof
    if not analysis_has_dates: asof = None
    active = {}
    root = v.get('root_cause')
    for item in ([root] if isinstance(root, dict) else []) + list(details.get('secondary_failures') or []):
        if isinstance(item, dict) and item.get('code'): active[str(item['code'])] = item
    for code in engine.get('active_fm_codes') or []: active.setdefault(str(code), {'code': code})
    drivers = []
    def driver(id, group, priority, topic, tone, role, title, detail, explanation=None, material=False):
        drivers.append(dict(id=id, group=group, priority=priority, topic=topic, tone=tone, role=role, title=title,
                            detail=str(detail or ''), explanation=str(explanation or detail or ''), link='Open the evidence', material=material))
    for code, item in active.items():
        label, title = FM.get(code, (str(item.get('label') or code), 'A diagnostic concern is reported.'))
        driver(code, 'diagnostics', 95 + (_num(item.get('severity')) or 0) / 100, 'structure', 'negative', 'Relationship concern', title, item.get('summary') or item.get('label') or label)
    if not active: driver('structure', 'diagnostics', 95, 'structure', 'positive' if state == 'VALID' else 'neutral', 'Relationship assessment', health + '.', v.get('summary') or 'Detailed diagnostics unavailable.')
    backdrop = str(regime.get('current_regime') or 'Not supplied')
    driver('regime', 'timing', 75, 'timing', 'neutral', 'Market backdrop', backdrop.replace('_', ' ').capitalize() + ' backdrop.', 'Rolling regime read on the portfolio path. Context for the move, not an instruction.', regime.get('strategy') or regime.get('strategic_signal') or '')
    competing = attr.get('competing_causes') or details.get('competing_causes') or []
    counter = attr.get('counterfactuals') or details.get('counterfactuals') or []
    sensitive = [c for c in counter if isinstance(c, dict) and c.get('changes_conclusion') is True]
    if sensitive: driver('robustness', 'robustness', 110, 'all', 'negative', 'Evidence sensitivity', 'An alternative check changes the conclusion.', sensitive[0].get('result') or sensitive[0].get('test'), material=True)
    if engine.get('path') in ('fallback_heuristic', 'error'): driver('coverage', 'coverage', 120, 'coverage', 'negative', 'Analysis limitation', 'The full validity analysis was unavailable.', 'This report uses a fallback or incomplete diagnosis.', material=True)
    fund_text = _plain(claude_fs_html) if is_equity_pair else ''
    stance = _stance(memo_text, fund_text, is_equity_pair)
    if fund_text or (is_equity_pair and funds):
        _ft = {'for':('positive','Fundamentals support the position.'), 'against':('negative','Fundamentals lean against the position.'), 'neutral':('neutral','Fundamentals give no edge either way.')}.get(stance['state'], ('neutral','Review the relative business case.'))
        driver('fundamentals', 'fundamentals', 70 if stance['state'] != 'against' else 100, 'fundamentals', _ft[0], 'Business case', _ft[1], ('Conviction ' + stance['conviction'].lower() + '. ' if stance.get('conviction') else '') + 'Earnings, valuation and analyst evidence are in the research.', 'The fundamental stance is combined with the engine view to produce the decision; it does not replace it.')
    conditions = [
        {'title':'A change in relationship health', 'detail':'Reassess diagnostic concerns and robustness checks.', 'topic':'structure'},
        {'title':'A change in market support', 'detail':'Review the regime, timing and whether the position still fits.', 'topic':'timing'},
        {'title':'A payoff that compensates for costs and risk', 'detail':'Include trading costs, financing and the intended holding period.', 'topic':'thesis'}]
    if is_equity_pair and fund_text: conditions[1] = {'title':'A change in the relative business case', 'detail':'Review results, guidance and catalysts in the research.', 'topic':'fundamentals'}
    signals = []
    for code, (label, question) in FM.items():
        item = active.get(code)
        signals.append({'id':code, 'question':question, 'name':label, 'available':item is not None,
                        'summary': str(item.get('summary') or item.get('label') or label) + (f" Severity: {item['severity']}/100." if _num(item.get('severity')) is not None else '') if item else '',
                        'missingLabel':'Not reported active; individual test evidence is not included.'})
    gross = _num(position_gross_exposure)
    risk = {'basis':'gross exposure' if gross and gross > 0 else None, 'basisConfirmed':bool(gross and gross > 0), 'positionAmount':gross, 'currency':'USD', 'horizonDays':risk_horizon_days if analysis_has_dates else None,
            'summary':'Public-asset returns are daily P&L divided by gross exposure. These historical estimates exclude costs not included by the engine.'}
    for output, key in [('var95Percent','var_95'), ('expectedShortfall95Percent','cvar_95'), ('maxDrawdownPercent','max_drawdown')]:
        n = _num(stats.get(key)); risk[output] = max(0, -n * 100) if n is not None else None
    # ── Four inputs and the single decision verb ──
    z_last = series[-1]['value'] if series else _num(regime.get('z_score')) if not hasattr(regime.get('z_score'), 'iloc') else None
    adf_p = _num(regime.get('adf_pvalue'))
    total_ret, max_dd = _num(stats.get('total_return')), _num(stats.get('max_drawdown'))
    dd_pct = abs(max_dd) * 100 if max_dd is not None else None
    ret_pct = total_ret * 100 if total_ret is not None else None
    if adf_p is not None and adf_p > 0.8:
        stats_in = {'state':'against', 'label':'Not supportive', 'tone':'negative', 'detail':'The spread is trending rather than reverting; the statistical case for a bounce is not there.'}
    elif z_last is None:
        stats_in = {'state':'unknown', 'label':'Not assessed', 'tone':'neutral', 'detail':'No dated path is available to place the spread against its reference.'}
    elif z_last <= -0.75:
        stats_in = {'state':'for', 'label':'Supportive', 'tone':'positive', 'detail':f'The spread is about {abs(z_last):.1f}σ against the position; a partial recovery is the statistical expectation.' + (' Beyond 2σ, the setup is unusually stretched.' if z_last <= -2 else '')}
    elif z_last >= 0.75:
        stats_in = {'state':'against', 'label':'Extended', 'tone':'negative', 'detail':f'The spread is about {abs(z_last):.1f}σ ahead of the position; reversion would give back gains.'}
    else:
        stats_in = {'state':'neutral', 'label':'Neutral', 'tone':'neutral', 'detail':f'The spread sits near its reference ({z_last:+.1f}σ); no statistical pull either way.'}
    _conv = (' (conviction ' + stance['conviction'].lower() + ')') if stance.get('conviction') else ''
    fund_in = {'n/a':{'state':'n/a','label':'Not applicable','tone':'neutral','detail':'Fundamental research is only run for equity pairs.'},
               'for':{'state':'for','label':'Supportive'+_conv,'tone':'positive','detail':'The research favours the long leg over the short leg.'},
               'against':{'state':'against','label':'Against'+_conv,'tone':'negative','detail':'The research favours the short leg, arguing against the position as held.'},
               'neutral':{'state':'neutral','label':'Neutral'+_conv,'tone':'neutral','detail':'The research gives no fundamental edge either way.'},
               'unknown':{'state':'unknown','label':'Not assessed','tone':'neutral','detail':'No fundamental stance could be read from the research.'}}[stance['state']]
    if dd_pct is None and ret_pct is None:
        risk_in = {'state':'unknown', 'label':'Not assessed', 'tone':'neutral', 'detail':'Realised return and drawdown were not supplied.'}
    elif (dd_pct or 0) >= 15 or (ret_pct is not None and ret_pct <= -10):
        risk_in = {'state':'stretched', 'label':'Stretched', 'tone':'negative', 'detail':f'The trade is {ret_pct:+.0f}% over the sample with a {dd_pct:.0f}% peak-to-trough drawdown; a healthy relationship can still be an expensive position.'}
    elif (dd_pct or 0) >= 8:
        risk_in = {'state':'elevated', 'label':'Elevated', 'tone':'neutral', 'detail':f'{ret_pct:+.0f}% over the sample, {dd_pct:.0f}% maximum drawdown.'}
    else:
        risk_in = {'state':'contained', 'label':'Contained', 'tone':'positive', 'detail':f'{ret_pct:+.0f}% over the sample, {dd_pct:.0f}% maximum drawdown.'}
    health_in = {'VALID':{'state':'strong','label':'Healthy','tone':'positive','detail':'The pair is behaving as it has historically.'},
                 'DEGRADED':{'state':'strained','label':'Under strain','tone':'negative','detail':'Diagnostics show the relationship weakening.'},
                 'BROKEN':{'state':'broken','label':'Broken','tone':'broken','detail':'The relationship has stopped behaving as expected.'},
                 'INVALID':{'state':'broken','label':'Broken','tone':'broken','detail':'The relationship has stopped behaving as expected.'}}.get(state, {'state':'unknown','label':'Not assessed','tone':'neutral','detail':'Relationship diagnostics were not supplied.'})
    if active and state == 'VALID':
        _first = next(iter(active.values()))
        health_in['detail'] = 'Behaving normally. One thing has changed: ' + str(_first.get('summary') or FM.get(next(iter(active)), ('a diagnostic', ''))[0]).split('.')[0].lower() + '.'
    verb, idea_verb, rule = _decide(action, state, stats_in, fund_in, risk_in, adf_p, z_last)
    inputs = [dict(key='health', name='Relationship', topic='structure', **health_in),
              dict(key='statistics', name='Statistics', topic='chart', **stats_in),
              dict(key='fundamentals', name='Fundamentals', topic='fundamentals', **fund_in),
              dict(key='risk', name='Realised risk', topic='risk', **risk_in)]
    # Plain-English "why" paragraph: the four inputs reconciled, no engine vocabulary.
    why = []
    why.append({'strong':'The pair is behaving normally', 'strained':'The relationship is under strain', 'broken':'The relationship has broken'}.get(health_in['state'], 'Relationship health is not assessed'))
    if stats_in['state'] == 'for': why[-1] += f', and the spread is about {abs(z_last):.1f}σ against the position, so a partial recovery is the statistical expectation.'
    elif stats_in['state'] == 'against' and z_last is not None and z_last >= 0.75: why[-1] += f', but the spread is about {abs(z_last):.1f}σ ahead of the position and reversion would give back gains.'
    elif stats_in['state'] == 'against': why[-1] += ', but the spread is trending rather than reverting.'
    else: why[-1] += ', with the spread near its reference.'
    if fund_in['state'] in ('for','against','neutral'):
        why.append({'for':'Fundamentals support the position', 'against':'Fundamentals lean the other way', 'neutral':'Fundamentals give no edge either way'}[fund_in['state']] + (f" with {stance['conviction'].lower()} conviction." if stance.get('conviction') else '.'))
    if risk_in['state'] == 'stretched': why.append(f'The trade is already {abs(ret_pct):.0f}% {"down" if ret_pct < 0 else "up"} with a {dd_pct:.0f}% drawdown.')
    elif risk_in['state'] in ('elevated','contained') and ret_pct is not None: why.append(f'Realised risk is {risk_in["state"]}: {ret_pct:+.0f}% over the sample, {dd_pct:.0f}% maximum drawdown.')
    why.append({'Hold':'Nothing argues for adding and nothing argues for leaving.', 'Reduce':'Enough is pulling against the position to take some off.', 'Exit':'The case for holding no longer holds together.', 'Build':'The setup is as strong as this relationship offers.'}[verb])
    verdict = {'verb':verb, 'ideaVerb':idea_verb, 'tone':{'Build':'positive','Hold':'neutral','Reduce':'negative','Exit':'broken'}[verb], 'summary':' '.join(why), 'rule':rule, 'engineDecision':action, 'engineRationale':rationale}
    next_review = _next_catalyst(memo_text, fund_text, after=asof)
    report = dict(
        schemaVersion=1, id=portfolio_name, portfolioName=portfolio_name, asOf=asof, illustrative=False, hasDates=bool(analysis_has_dates), sector=portfolio_name,
        assets={'long':asset(longs,'Long'), 'short':asset(shorts,'Short')}, hasLong=bool(longs), hasShort=bool(shorts),
        validity={'status':state, 'label':health, 'score':score, 'summary':v.get('summary') or 'No summary supplied.', 'tests':[
            {'label':'Diagnosis confidence', 'value':str(v.get('confidence')) if v.get('confidence') is not None else 'Not supplied'},
            {'label':'Mean-reversion half-life', 'value':str(regime.get('halflife'))+' observations' if _num(regime.get('halflife')) is not None else 'Not supplied'}]},
        verdict=verdict, inputs=inputs,
        position={'headline':[verb], 'summary':verdict['summary'], 'label':'Decision: '+verb, 'assessment':verb, 'tone':verdict['tone'], 'basis':'Decision'},
        newIdea={'headline':[idea_verb], 'summary':('The relationship is healthy and the spread is stretched against this direction with nothing arguing against it.' if idea_verb=='Build' else 'A new position needs a healthy relationship, a spread beyond 2σ in its favour and fundamentals that do not argue against it. That bar is not met today.') + ' Net payoff after costs is not modelled; check it before committing capital.', 'label':'New idea: '+idea_verb, 'assessment':idea_verb, 'tone':'positive' if idea_verb=='Build' else 'neutral'},
        opportunity={'assessed':False, 'summary':'Net payoff not modelled', 'detail':'The engine supplies a position decision, not a calibrated payoff forecast after trading, financing and borrowing costs. Historical returns and spread distance do not substitute for that forecast.'},
        drivers=drivers, conditions=conditions, risk=risk, nextReview=next_review,
        regime={'available':bool(regime), 'label':backdrop, 'summary':str(regime.get('strategy') or '')}, signals=signals,
        robustness={'competingExplanations':[{'summary':str(c.get('label',''))+': '+str(c.get('evidence','')), **c} for c in competing if isinstance(c,dict)],
                    'counterfactuals':[{'summary':str(c.get('test','Check'))+': '+str(c.get('result',''))+(' — changes the conclusion.' if c.get('changes_conclusion') else ''), **c} for c in counter if isinstance(c,dict)],
                    'dependencies':details.get('dependencies') or raw.get('dependencies') or [], 'trustAdjustment':details.get('trust_penalty') if details.get('trust_penalty') is not None else raw.get('trust_penalty')},
        chart={'title':'How unusual is the current move?', 'description':'Portfolio value relative to its rolling reference', 'label':'Normalized portfolio path', 'illustrative':False,
               'reference':{'lower':-2,'upper':2} if series else None, 'series':series, 'events':events, 'snapshots':[],
               'definition':f'The engine’s rolling z-score of cumulative portfolio value: (value − rolling mean) / rolling standard deviation, using {chart_window} observations. This is not an estimated cointegrating residual. Warm-up and zero-variance observations are omitted. The ±2 band is a statistical reference, not a trade rule or proof of validity.'},
        research={'trade':str(memo_text or ''), 'fundamentals':fund_text, 'metrics':funds if is_equity_pair else {}},
        diagnostics={'validity':raw, 'regime':regime, 'statistics':stats, 'decision':dd, 'positions':{'long_weights':longs,'short_weights':shorts,'gross_exposure':gross}, 'analysis_observations':analysis_period_days},
        referenceCharts=[], sources=[{'label':'Bavella analysis engine','date':asof,'detail':'Calculated from submitted exposure and available observations. Position decision and validity are preserved separately.'}],
        coverage=['Historical chart: '+(str(len(series))+' valid dated observations.' if series else 'No valid dated z-score history.'),
                  'Historical validity snapshots are not stored by this route; chart annotations are rolling regime changes.',
                  'Fundamental research: '+('available; review its source dates.' if fund_text else 'not included.'),
                  'Net expected payoff after costs is not calculated.', 'Engine path: '+str(engine.get('path') or 'not supplied')])
    for title, image in [('Regime reference',regime_chart_base64), ('Historical performance',performance_chart_base64), ('Return distribution',distribution_chart_base64)]:
        if isinstance(image,str) and re.fullmatch(r'[A-Za-z0-9+/=\s]+',image): report['referenceCharts'].append({'title':title,'base64':image})
    report = _clean(report)
    template = Path(__file__).with_name('report_template.html').read_text(encoding='utf-8')
    payload = json.dumps(report,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    # Split before inserting untrusted strings: inserted content is never re-templated.
    before, after = template.split('/*__REPORT__*/')
    before = before.replace('<!--__PRINT_REPORT__-->', _print_report(report))
    return before + payload + after


def _print_report(report):
    """No-JavaScript fallback for the existing WeasyPrint export route."""
    e = lambda v: escape(str(v if v is not None else 'Not supplied'))
    out = ['<section class="print-report"><h1>'+e(report['portfolioName'])+'</h1><p>As of '+e(report['asOf'] or 'undated observations')+'</p>',
           '<h2>'+e(report['verdict']['verb'])+'</h2><p>'+e(report['verdict']['summary'])+'</p>',
           '<p>'+' · '.join(e(i['name'])+': '+e(i['label']) for i in report['inputs'])+'</p>']
    points = report['chart']['series']
    if points:
        lo, hi = min(-2,min(p['value'] for p in points))-.3, max(2,max(p['value'] for p in points))+.3
        first = datetime.strptime(points[0]['date'],'%Y-%m-%d')
        span = max(1,(datetime.strptime(points[-1]['date'],'%Y-%m-%d')-first).days)
        x = lambda p: 55+625*(datetime.strptime(p['date'],'%Y-%m-%d')-first).days/span
        y = lambda v: 230-(v-lo)/(hi-lo)*200
        path = ' '.join(('M' if i==0 else 'L')+f'{x(p):.2f},{y(p["value"]):.2f}' for i,p in enumerate(points))
        out.append(f'<h3>Normalized portfolio path</h3><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 270" width="100%"><rect x="55" y="{y(2):.2f}" width="625" height="{y(-2)-y(2):.2f}" fill="#eff4fc"/><path d="{path}" fill="none" stroke="#345fb0" stroke-width="2"/>')
        for val in [-2,0,2]: out.append(f'<text x="8" y="{y(val):.2f}" font-size="12">{val}σ</text>')
        out.append('<text x="55" y="258" font-size="12">'+e(points[0]['date'])+'</text><text x="680" y="258" text-anchor="end" font-size="12">'+e(points[-1]['date'])+'</text></svg>')
    out.append('<h3>Evidence</h3>')
    for d in report['drivers']: out.append('<p><strong>'+e(d['title'])+'</strong> '+e(d['detail'])+'</p>')
    out.append('<h3>Reassessment conditions</h3><ul>')
    for c in report['conditions']: out.append('<li>'+e(c['title'])+': '+e(c['detail'])+'</li>')
    out.append('</ul><h3>Risk reference</h3>')
    r=report['risk']
    for title,key in [('95% loss threshold','var95Percent'),('Average beyond threshold','expectedShortfall95Percent')]:
        if r.get(key) is not None:
            amount=f" / USD {r[key]*r['positionAmount']/100:,.0f}" if r['basisConfirmed'] and r.get('horizonDays')==1 else ''
            out.append('<p>'+title+': '+e(f"{r[key]:.2f}%")+e(amount)+'. Basis: '+e(r['basis'] or 'not confirmed')+'</p>')
    out.append('<p>Historical estimates; losses can be larger. No net payoff forecast is supplied.</p>')
    for title,key in [('Trade research','trade'),('Fundamental research','fundamentals')]:
        if report['research'][key]: out.append('<h3>'+title+'</h3><div style="white-space:pre-wrap">'+e(report['research'][key])+'</div>')
    out.append('</section>')
    return ''.join(out)
