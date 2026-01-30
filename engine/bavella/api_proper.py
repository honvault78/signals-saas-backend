"""
Bavella v2 — Validity-Enforced API Layer
=========================================

THIS API MAKES IT IMPOSSIBLE TO LEAK SUPPRESSED OUTPUTS.

The enforcement happens at the serialization layer, not at the UI layer.
There is no "full response" that gets filtered later.
The response is CONSTRUCTED by the Governor.

Design principles:
    1. Governor controls what gets serialized
    2. No endpoint returns unvalidated data
    3. INVALID nodes produce validity info ONLY
    4. All responses include validity metadata
    5. Export endpoints respect suppression

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json

from .core import (
    ValidityState, ValidityVerdict, FailureMode, FailureSignal,
    GovernedValue, GovernorRefusal, AnalysisNode
)
from .governor import (
    Governor, OutputLevel, OutputRequest, OutputDecision,
    GovernedResponseBuilder, GovernedExporter
)


# =============================================================================
# RESPONSE ENVELOPE (always includes validity)
# =============================================================================

@dataclass
class APIResponse:
    """
    Standard API response envelope.
    
    EVERY response includes validity metadata.
    The 'data' field is ONLY populated if Governor permits.
    """
    # Always present (L0)
    success: bool
    timestamp: str
    request_id: str
    
    # Validity info (always present - L0)
    validity: Dict[str, Any]
    
    # Data (ONLY if permitted by Governor)
    data: Optional[Dict[str, Any]] = None
    
    # Suppression info (present if anything was suppressed)
    suppressed: Optional[List[str]] = None
    suppression_reason: Optional[str] = None
    
    # Watermark (present if data is DEGRADED)
    watermark: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "validity": self.validity,
        }
        
        if self.data is not None:
            result["data"] = self.data
        
        if self.suppressed:
            result["suppressed_fields"] = self.suppressed
            result["suppression_reason"] = self.suppression_reason
        
        if self.watermark:
            result["watermark"] = self.watermark
        
        return result
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# =============================================================================
# GOVERNED SERIALIZER (the key enforcement layer)
# =============================================================================

class GovernedSerializer:
    """
    Serializes analysis results with Governor enforcement.
    
    THIS IS WHERE SUPPRESSION BECOMES STRUCTURAL.
    
    The serializer doesn't receive data and then filter it.
    It ASKS the Governor what to include, field by field.
    """
    
    def __init__(self, governor: Governor):
        self._governor = governor
    
    def serialize_analysis(
        self,
        node_id: str,
        verdict: ValidityVerdict,
        analysis_data: Dict[str, Any],
        request_id: str,
    ) -> APIResponse:
        """
        Serialize an analysis result.
        
        The Governor decides what fields appear in the response.
        """
        # Register verdict with Governor
        self._governor.register_verdict(node_id, verdict)
        
        # Build response using Governor
        builder = GovernedResponseBuilder(self._governor)
        
        # Validity info is always L0 (always included)
        builder.add_always("score", node_id, verdict.score)
        builder.add_always("state", node_id, verdict.state.value)
        builder.add_always("attributions", node_id, [
            {"failure_mode": fm.name, "contribution_pct": pct}
            for fm, pct in verdict.attributions
        ])
        
        if verdict.is_killed:
            builder.add_always("killed_by", node_id, verdict.killed_by.name)
            builder.add_always("kill_reason", node_id, verdict.kill_reason)
        
        validity_data = builder.build()
        
        # Now try to add analysis data (L1 = metrics, L2 = predictions)
        data_builder = GovernedResponseBuilder(self._governor)
        suppressed_fields = []
        
        for key, value in analysis_data.items():
            # Determine output level based on field type
            level = self._classify_field(key)
            
            if level == OutputLevel.L1_VALID_OR_DEGRADED:
                data_builder.add_metric(key, node_id, value)
            elif level == OutputLevel.L2_VALID_ONLY:
                data_builder.add_prediction(key, node_id, value)
            else:
                data_builder.add_always(key, node_id, value)
        
        data_result = data_builder.build()
        
        # Check for suppressed fields
        suppressed = data_result.pop("_suppressed", None)
        suppression_note = data_result.pop("_suppression_note", None)
        
        # Check for watermark
        watermark = None
        if verdict.state == ValidityState.DEGRADED:
            watermark = (
                f"⚠️ VALIDITY DEGRADED ({verdict.score:.0f}/100) - "
                f"Results should be interpreted with caution"
            )
            if verdict.dominant_failure:
                watermark += f" - Primary concern: {verdict.dominant_failure.name}"
        
        return APIResponse(
            success=True,
            timestamp=datetime.utcnow().isoformat(),
            request_id=request_id,
            validity=validity_data,
            data=data_result if data_result else None,
            suppressed=suppressed,
            suppression_reason=suppression_note,
            watermark=watermark,
        )
    
    def _classify_field(self, field_name: str) -> OutputLevel:
        """Classify a field into an output level."""
        # L2 fields (VALID only)
        l2_patterns = [
            "prediction", "forecast", "recommendation", "signal",
            "action", "trade", "position", "allocation"
        ]
        
        # L0 fields (always)
        l0_patterns = [
            "id", "timestamp", "series_id", "node_id", "type"
        ]
        
        name_lower = field_name.lower()
        
        for pattern in l0_patterns:
            if pattern in name_lower:
                return OutputLevel.L0_ALWAYS
        
        for pattern in l2_patterns:
            if pattern in name_lower:
                return OutputLevel.L2_VALID_ONLY
        
        # Default to L1 (metrics)
        return OutputLevel.L1_VALID_OR_DEGRADED


# =============================================================================
# API ENDPOINT HANDLERS (using FastAPI-like patterns)
# =============================================================================

class ValidityAwareAPI:
    """
    API handlers that enforce validity at every endpoint.
    
    Can be mounted to FastAPI, Flask, or any framework.
    """
    
    def __init__(self):
        self._governor = Governor()
        self._serializer = GovernedSerializer(self._governor)
    
    def new_session(self) -> str:
        """Start a new Governor session."""
        self._governor = Governor()
        self._serializer = GovernedSerializer(self._governor)
        return self._governor.session_id
    
    # =========================================================================
    # ANALYSIS ENDPOINT
    # =========================================================================
    
    def handle_analyze(
        self,
        request_data: Dict[str, Any],
        request_id: str,
    ) -> APIResponse:
        """
        Handle POST /v2/analyze
        
        Returns validity-governed analysis.
        """
        # This would be called after running the actual analysis
        # For now, assume we have results
        
        node_id = request_data.get("node_id", "analysis")
        verdict = request_data.get("verdict")
        analysis_data = request_data.get("analysis_data", {})
        
        if verdict is None:
            return APIResponse(
                success=False,
                timestamp=datetime.utcnow().isoformat(),
                request_id=request_id,
                validity={"error": "No verdict provided"},
            )
        
        return self._serializer.serialize_analysis(
            node_id=node_id,
            verdict=verdict,
            analysis_data=analysis_data,
            request_id=request_id,
        )
    
    # =========================================================================
    # VALIDITY-ONLY ENDPOINT
    # =========================================================================
    
    def handle_validity(
        self,
        request_data: Dict[str, Any],
        request_id: str,
    ) -> APIResponse:
        """
        Handle POST /v2/validity
        
        Returns ONLY validity information (always permitted).
        """
        node_id = request_data.get("node_id", "validity")
        verdict = request_data.get("verdict")
        
        if verdict is None:
            return APIResponse(
                success=False,
                timestamp=datetime.utcnow().isoformat(),
                request_id=request_id,
                validity={"error": "No verdict provided"},
            )
        
        # Register verdict
        self._governor.register_verdict(node_id, verdict)
        
        # Validity is always L0
        validity_data = {
            "score": verdict.score,
            "state": verdict.state.value,
            "attributions": [
                {"failure_mode": fm.name, "contribution_pct": pct}
                for fm, pct in verdict.attributions
            ],
        }
        
        if verdict.is_killed:
            validity_data["killed_by"] = verdict.killed_by.name
            validity_data["kill_reason"] = verdict.kill_reason
        
        if verdict.inherited_from:
            validity_data["inherited_from"] = verdict.inherited_from
        
        return APIResponse(
            success=True,
            timestamp=datetime.utcnow().isoformat(),
            request_id=request_id,
            validity=validity_data,
            data=None,  # This endpoint only returns validity
        )
    
    # =========================================================================
    # EXPORT ENDPOINT (strictly governed)
    # =========================================================================
    
    def handle_export(
        self,
        request_data: Dict[str, Any],
        request_id: str,
    ) -> Union[APIResponse, str]:
        """
        Handle POST /v2/export
        
        Exports ONLY if VALID. Returns APIResponse with error if not.
        """
        node_id = request_data.get("node_id")
        verdict = request_data.get("verdict")
        export_data = request_data.get("data", {})
        format = request_data.get("format", "json")
        
        if verdict is None:
            return APIResponse(
                success=False,
                timestamp=datetime.utcnow().isoformat(),
                request_id=request_id,
                validity={"error": "No verdict provided"},
            )
        
        # Register verdict
        self._governor.register_verdict(node_id, verdict)
        
        # Check export permission
        exporter = GovernedExporter(self._governor)
        
        if not exporter.can_export(node_id):
            return APIResponse(
                success=False,
                timestamp=datetime.utcnow().isoformat(),
                request_id=request_id,
                validity={
                    "score": verdict.score,
                    "state": verdict.state.value,
                },
                data=None,
                suppressed=["full_export"],
                suppression_reason=(
                    f"Export not permitted: validity state is {verdict.state.value}. "
                    "Full exports require VALID state."
                ),
            )
        
        # Export permitted
        exported = exporter.export_with_governance(node_id, export_data, format)
        
        if format == "json":
            # Return the JSON string directly
            return exported
        
        return APIResponse(
            success=True,
            timestamp=datetime.utcnow().isoformat(),
            request_id=request_id,
            validity={
                "score": verdict.score,
                "state": verdict.state.value,
            },
            data={"export": "success", "format": format},
        )
    
    # =========================================================================
    # AUDIT ENDPOINT
    # =========================================================================
    
    def handle_audit(self, request_id: str) -> APIResponse:
        """
        Handle GET /v2/audit
        
        Returns Governor emission log.
        """
        summary = self._governor.summary()
        log = self._governor.get_emission_log()
        
        return APIResponse(
            success=True,
            timestamp=datetime.utcnow().isoformat(),
            request_id=request_id,
            validity={"audit_complete": True},
            data={
                "session_id": summary["session_id"],
                "total_requests": summary["total_requests"],
                "permitted": summary["permitted"],
                "refused": summary["refused"],
                "refusal_rate": summary["refusal_rate"],
                "emissions": log[-100:],  # Last 100 emissions
            },
        )


# =============================================================================
# FASTAPI INTEGRATION (if FastAPI is available)
# =============================================================================

def create_fastapi_app():
    """
    Create a FastAPI app with validity-enforced endpoints.
    
    Usage:
        app = create_fastapi_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
        from typing import Optional, List, Dict, Any
    except ImportError:
        raise ImportError("FastAPI not installed. pip install fastapi uvicorn")
    
    app = FastAPI(
        title="Bavella v2 API",
        description="Validity-enforced epistemic infrastructure",
        version="2.1.0",
    )
    
    api = ValidityAwareAPI()
    
    # =========================================================================
    # MIDDLEWARE: Add request ID to all requests
    # =========================================================================
    
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        import uuid
        request.state.request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response
    
    # =========================================================================
    # REQUEST/RESPONSE MODELS
    # =========================================================================
    
    class AnalyzeRequest(BaseModel):
        series_values: List[float]
        series_timestamps: Optional[List[str]] = None
        series_id: str = "series"
        include_metrics: bool = True
        include_predictions: bool = True
    
    class ValidityResponse(BaseModel):
        success: bool
        timestamp: str
        request_id: str
        validity: Dict[str, Any]
        data: Optional[Dict[str, Any]] = None
        suppressed_fields: Optional[List[str]] = None
        suppression_reason: Optional[str] = None
        watermark: Optional[str] = None
    
    # =========================================================================
    # ENDPOINTS
    # =========================================================================
    
    @app.get("/v2/health")
    async def health():
        return {
            "status": "healthy",
            "version": "2.1.0",
            "governance": "active",
            "session_id": api._governor.session_id,
        }
    
    @app.post("/v2/analyze", response_model=ValidityResponse)
    async def analyze(request: AnalyzeRequest, req: Request):
        """
        Analyze a time series and return validity-governed results.
        
        Suppression is enforced at the serialization layer.
        """
        # In real implementation, this would:
        # 1. Convert request to Series
        # 2. Run NIS transform
        # 3. Run detectors
        # 4. Compute validity
        # 5. Return governed response
        
        # For now, return placeholder showing governance works
        from .core import ValidityVerdict, ValidityState
        
        # Simulate a verdict (in real code, this comes from analysis)
        mock_verdict = ValidityVerdict(
            score=75.0,
            state=ValidityState.VALID,
        )
        
        response = api.handle_analyze(
            request_data={
                "node_id": request.series_id,
                "verdict": mock_verdict,
                "analysis_data": {
                    "mean": 100.0,
                    "std": 15.0,
                    "prediction": [101, 102, 103],
                },
            },
            request_id=req.state.request_id,
        )
        
        return response.to_dict()
    
    @app.post("/v2/validity", response_model=ValidityResponse)
    async def validity(request: AnalyzeRequest, req: Request):
        """
        Get validity assessment only (always permitted).
        """
        from .core import ValidityVerdict, ValidityState
        
        mock_verdict = ValidityVerdict(
            score=75.0,
            state=ValidityState.VALID,
        )
        
        response = api.handle_validity(
            request_data={
                "node_id": request.series_id,
                "verdict": mock_verdict,
            },
            request_id=req.state.request_id,
        )
        
        return response.to_dict()
    
    @app.get("/v2/audit", response_model=ValidityResponse)
    async def audit(req: Request):
        """
        Get Governor audit log.
        """
        response = api.handle_audit(req.state.request_id)
        return response.to_dict()
    
    @app.post("/v2/session")
    async def new_session():
        """
        Start a new Governor session.
        """
        session_id = api.new_session()
        return {"session_id": session_id}
    
    return app


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Run the validity-enforced API server."""
    try:
        import uvicorn
        app = create_fastapi_app()
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        print("To run the API server, install: pip install fastapi uvicorn")
        raise
