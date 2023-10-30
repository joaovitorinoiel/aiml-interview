import {
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  Box,
  Typography,
  Modal,
} from "@mui/material";
import { useCallback, useEffect, useState } from "react";
import { useAuth } from "../hooks/useAuth";
import VideoLibraryIcon from "@mui/icons-material/VideoLibrary";
import RefreshIcon from "@mui/icons-material/Refresh";
import api from "../services/api";
import { Assessment } from "@mui/icons-material";

const columns = [
  { id: "date", label: "Data", minWidth: 100 },
  { id: "duration", label: "Duração", minWidth: 100 },
  { id: "video", label: "Vídeo", minWidth: 100 },
  { id: "report", label: "Relatório", minWidth: 100 },
];

const modalStyle = {
  position: "absolute",
  top: "50%",
  left: "50%",
  transform: "translate(-50%, -50%)",
  width: "50%",
  bgcolor: "background.paper",
  boxShadow: 24,
  p: 4,
};

function RecordsTable() {
  const { user } = useAuth();
  const [records, setRecords] = useState([]);
  const [reportModal, setReportModal] = useState(false);
  const [metrics, setMetrics] = useState({
    transcription: "",
    feedback_bedrock: "",
  });

  const handleS3Link = (file) => {
    api
      .get("download", {
        params: { filename: file },
      })
      .then((response) => {
        window.open(response.data["url"]);
      });
  };

  const handleOpenReport = (record) => {
    setMetrics(JSON.parse(record.report.replace(/'/g, '"')));
    setReportModal(true);
  };

  const handleCloseReport = () => setReportModal(false);

  const refreshTable = useCallback((email) => {
    api
      .get("records", {
        params: { email: email },
      })
      .then((response) => {
        setRecords(response.data.results);
      });
  }, []);

  useEffect(() => {
    refreshTable(user["userEmail"]);
  }, [refreshTable, user]);

  return (
    <Paper sx={{ width: "100%", overflow: "hidden" }}>
      <Box
        sx={{
          display: "flex",
          flexDirection: "row-reverse",
        }}
      >
        <Button
          onClick={() => {
            refreshTable(user["userEmail"]);
          }}
        >
          <RefreshIcon />
        </Button>
      </Box>
      <TableContainer sx={{ maxHeight: 440 }}>
        <Table stickyHeader aria-label="sticky table">
          <TableHead>
            <TableRow>
              {columns.map((column) => (
                <TableCell
                  key={column.id}
                  align={column.align}
                  style={{ minWidth: column.minWidth }}
                >
                  {column.label}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {records.map((row) => (
              <TableRow key={row.record_id}>
                {columns.map((column) => {
                  const value = row[column.id];
                  if (column.id === "video") {
                    return (
                      <TableCell key={column.id} align={column.align}>
                        {value !== "" ? (
                          <Button
                            onClick={() => {
                              handleS3Link(value);
                            }}
                          >
                            <VideoLibraryIcon />
                          </Button>
                        ) : (
                          <Typography>aguardando</Typography>
                        )}
                      </TableCell>
                    );
                  } else if (column.id === "report") {
                    return (
                      <TableCell key={column.id} align={column.align}>
                        {value !== "" ? (
                          <Button
                            onClick={() => {
                              handleOpenReport(row);
                            }}
                          >
                            <Assessment />
                          </Button>
                        ) : (
                          <Typography>aguardando</Typography>
                        )}
                      </TableCell>
                    );
                  } else {
                    return (
                      <TableCell key={column.id} align={column.align}>
                        <Typography>
                          {value !== "" ? value : "aguardando"}
                        </Typography>
                      </TableCell>
                    );
                  }
                })}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <Modal
        open={reportModal}
        onClose={handleCloseReport}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
      >
        <Box sx={modalStyle}>
          <Typography id="modal-modal-title" variant="h4">
            Resultado da simulação
          </Typography>
          <Typography id="modal-modal-title" variant="h5" sx={{ mt: 2 }}>
            Feedback
          </Typography>
          <Typography id="modal-modal-description" sx={{ mt: 2 }}>
            {metrics.feedback_bedrock}
          </Typography>
          <Typography id="modal-modal-title" variant="h5" sx={{ mt: 2 }}>
            Transcrição
          </Typography>
          <Typography id="modal-modal-description" sx={{ mt: 2 }}>
            {metrics.transcription}
          </Typography>
        </Box>
      </Modal>
    </Paper>
  );
}

export default RecordsTable;
