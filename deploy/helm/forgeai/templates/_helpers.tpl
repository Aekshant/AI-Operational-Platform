{{- define "forgeai.labels" -}}
app.kubernetes.io/part-of: forgeai
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{- define "forgeai.secretName" -}}
{{- default (printf "%s-auth" .Release.Name) .Values.auth.existingSecret -}}
{{- end -}}

{{- define "forgeai.commonEnv" -}}
- name: FORGE_API_KEYS
  valueFrom: { secretKeyRef: { name: {{ include "forgeai.secretName" . }}, key: apiKeys } }
- name: FORGE_CONTROLPLANE_URL
  value: http://{{ .Release.Name }}-controlplane:8080
- name: FORGE_PROMETHEUS_URL
  value: {{ .Values.observability.prometheusUrl | quote }}
- name: FORGE_TEMPO_URL
  value: {{ .Values.observability.tempoUrl | quote }}
- name: FORGE_LOKI_URL
  value: {{ .Values.observability.lokiUrl | quote }}
- name: OTEL_EXPORTER_OTLP_ENDPOINT
  value: {{ .Values.observability.otlpEndpoint | quote }}
- name: OTEL_METRICS_EXPORTER
  value: none
- name: FORGE_TENANT_LIMITS
  value: {{ .Values.tenants | quote }}
- name: FORGE_SLO_TTFT_P95_S
  value: {{ .Values.slo.ttftP95Seconds | quote }}
- name: FORGE_SLO_LATENCY_P95_S
  value: {{ .Values.slo.latencyP95Seconds | quote }}
- name: FORGE_SLO_ERROR_RATE
  value: {{ .Values.slo.errorRate | quote }}
{{- end -}}
