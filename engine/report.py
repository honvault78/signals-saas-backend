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
            previous = None
            for index, row in regimes.iterrows():
                day, current = _date(index), str(row['regime'])
                if current.lower() in ('nan', 'none', 'unknown'):
                    previous = None; continue
                if previous is not None and current != previous and day and history[0]['date'] <= day <= history[-1]['date']:
                    events.append({'date': day, 'title': 'Market backdrop changed', 'topic': 'timing',
                                   'summary': f'Rolling regime changed from {previous} to {current}. This is a regime observation, not a historical validity or trade decision.'})
                previous = current
        return history, events
    except (AttributeError, TypeError, ValueError, KeyError): return [], []


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
    health = {'VALID':'Relationship intact', 'DEGRADED':'Relationship under strain', 'BROKEN':'Relationship broken', 'INVALID':'Relationship invalid'}.get(state, 'Relationship not assessed')
    verdict = {'EXIT':'The engine calls for an exit.', 'REDUCE':'The engine calls for less exposure.', 'WAIT':'The engine calls for waiting.', 'HOLD':'The engine supports holding.', 'ENTER':'The engine supports an entry.', 'REVERSE':'The engine calls for reversing direction.'}.get(action, 'The position needs assessment.')
    tone = 'negative' if action in ('EXIT', 'REDUCE', 'REVERSE') else 'neutral'
    action_text = action.capitalize() if action != 'NOT ASSESSED' else 'Not assessed'
    size = _num(dd.get('size_pct'))
    if size is not None and action in ('REDUCE', 'HOLD', 'REVERSE'): action_text += f' · {size:g}% target size'
    longs, shorts, funds = long_positions or {}, short_positions or {}, fundamental_data or {}
    def asset(positions, side):
        if len(positions) == 1:
            ticker = next(iter(positions))
            return {'name': str((funds.get(ticker) or {}).get('name') or ticker), 'ticker': ticker}
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
    driver('regime', 'timing', 75, 'timing', 'neutral', 'Market backdrop', backdrop.replace('_', ' ').capitalize() + ' backdrop.', regime.get('strategy') or regime.get('strategic_signal') or 'No strategy context supplied.')
    competing = attr.get('competing_causes') or details.get('competing_causes') or []
    counter = attr.get('counterfactuals') or details.get('counterfactuals') or []
    sensitive = [c for c in counter if isinstance(c, dict) and c.get('changes_conclusion') is True]
    if sensitive: driver('robustness', 'robustness', 110, 'all', 'negative', 'Evidence sensitivity', 'An alternative check changes the conclusion.', sensitive[0].get('result') or sensitive[0].get('test'), material=True)
    if engine.get('path') in ('fallback_heuristic', 'error'): driver('coverage', 'coverage', 120, 'coverage', 'negative', 'Analysis limitation', 'The full validity analysis was unavailable.', 'This report uses a fallback or incomplete diagnosis.', material=True)
    fund_text = _plain(claude_fs_html) if is_equity_pair else ''
    if fund_text or (is_equity_pair and funds): driver('fundamentals', 'fundamentals', 70, 'fundamentals', 'neutral', 'Business case', 'Review the relative business case.', 'Earnings, valuation and analyst evidence are available.', 'Fundamental research supports the analysis; it does not override the deterministic decision.')
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
    report = dict(
        schemaVersion=1, id=portfolio_name, portfolioName=portfolio_name, asOf=asof, illustrative=False, hasDates=bool(analysis_has_dates), sector=portfolio_name,
        assets={'long':asset(longs,'Long'), 'short':asset(shorts,'Short')}, hasLong=bool(longs), hasShort=bool(shorts),
        validity={'status':state, 'label':health, 'score':score, 'summary':v.get('summary') or 'No summary supplied.', 'tests':[
            {'label':'Diagnosis confidence', 'value':str(v.get('confidence')) if v.get('confidence') is not None else 'Not supplied'},
            {'label':'Mean-reversion half-life', 'value':str(regime.get('halflife'))+' observations' if _num(regime.get('halflife')) is not None else 'Not supplied'}]},
        position={'headline':[health+'.',verdict], 'summary':rationale, 'label':'Position view: '+action_text, 'assessment':action_text, 'tone':tone, 'basis':'Engine assessment'},
        newIdea={'headline':[health+'.','A new investment needs its own case.'], 'summary':'The engine assessment concerns the specified exposure. Review net payoff, direction and holding horizon before committing new capital.', 'label':'Net opportunity not assessed', 'assessment':'Assess net payoff', 'tone':'neutral'},
        opportunity={'assessed':False, 'summary':'Net payoff not modelled', 'detail':'The engine supplies a position decision, not a calibrated payoff forecast after trading, financing and borrowing costs. Historical returns and spread distance do not substitute for that forecast.'},
        drivers=drivers, conditions=conditions, risk=risk, nextReview=None,
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
           '<h2>'+e(' '.join(report['position']['headline']))+'</h2><p>'+e(report['position']['summary'])+'</p>',
           '<p>Engine assessment: '+e(report['position']['assessment'])+' · Validity: '+e(report['validity']['status'])+' / '+e(report['validity']['score'])+'</p>']
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
